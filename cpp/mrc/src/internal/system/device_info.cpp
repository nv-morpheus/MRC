/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "internal/system/device_info.hpp"

#include "mrc/cuda/common.hpp"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <glog/logging.h>
#include <hwloc.h>
#include <hwloc/bitmap.h>
#include <hwloc/linux.h>
#include <nvml.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>

namespace mrc::system {
struct NvmlHandle;
struct NvmlState;

#define TEST_BIT(_n, _p) (_n & (1UL << _p))

#define MRC_CHECK_NVML(expression)                                                           \
    {                                                                                        \
        auto __status = (expression);                                                        \
        if (__status != NVML_SUCCESS)                                                        \
        {                                                                                    \
            LOG(FATAL) << "NVML failed running '" << #expression                             \
                       << "'. Error msg: " << NvmlState::handle().nvmlErrorString(__status); \
        }                                                                                    \
    }

#define LOAD_NVTX_SYM(dll_ptr, function_var)                                                        \
    function_var = reinterpret_cast<decltype(function_var)>(dlsym(dll_ptr, #function_var));         \
    if (function_var == nullptr)                                                                    \
    {                                                                                               \
        throw std::runtime_error("Failed to load symbol " #function_var " from libnvidia-ml.so.1"); \
    }

/**
 * @brief This class wraps all calls to NVML and ensures that the library is dynamically loaded at runtime using
 * `dlopen`. This is necessary to remove the link dependency to `libnvidia-ml.so` which is a driver library. Because
 * driver libraries are loaded into the docker container via the NVIDIA container runtime, we do not want to directly
 * link to them. As well, if the driver is not found, we can still run MRC without GPUs. This gives us an oportunity to
 * catch the missing library instead of the OS throwing an error.
 *
 */
struct NvmlHandle
{
    NvmlHandle()
    {
        // First, try to open the library dynamically
        m_nvml_dll = dlopen("libnvidia-ml.so.1", RTLD_LOCAL | RTLD_LAZY);

        // Throw an error if missing
        if (m_nvml_dll == nullptr)
        {
            throw std::runtime_error("Could not open libnvidia-ml.so.1");
        }

        // Increment the dll counter since this was successfully loaded
        s_dll_counter++;

        // Now load the symbols. Keep init first
        LOAD_NVTX_SYM(m_nvml_dll, nvmlInit_v2);

        // Keep the rest of the functions sorted. Must match list of functions below!
        LOAD_NVTX_SYM(m_nvml_dll, nvmlDeviceGetCount_v2);
        LOAD_NVTX_SYM(m_nvml_dll, nvmlDeviceGetHandleByIndex_v2);
        LOAD_NVTX_SYM(m_nvml_dll, nvmlDeviceGetMemoryInfo);
        LOAD_NVTX_SYM(m_nvml_dll, nvmlDeviceGetMigMode);
        LOAD_NVTX_SYM(m_nvml_dll, nvmlDeviceGetName);
        LOAD_NVTX_SYM(m_nvml_dll, nvmlDeviceGetPciInfo_v3);
        LOAD_NVTX_SYM(m_nvml_dll, nvmlDeviceGetPowerManagementLimit);
        LOAD_NVTX_SYM(m_nvml_dll, nvmlDeviceGetPowerUsage);
        LOAD_NVTX_SYM(m_nvml_dll, nvmlDeviceGetTotalEnergyConsumption);
        LOAD_NVTX_SYM(m_nvml_dll, nvmlDeviceGetUUID);
        LOAD_NVTX_SYM(m_nvml_dll, nvmlErrorString);
        LOAD_NVTX_SYM(m_nvml_dll, nvmlShutdown);
    }

    ~NvmlHandle()
    {
        // Close the library on shutdown. Double check we only call dlclose when the last instance is being destroyed
        if (m_nvml_dll != nullptr && --s_dll_counter == 0)
        {
            dlclose(m_nvml_dll);
        }
    }

    // NOLINTBEGIN(readability-identifier-naming)
    // Keep the init function up top
    decltype(&::nvmlInit_v2) nvmlInit_v2{nullptr};

    // Keep the rest of the functions sorted. Add new ones as necessary
    decltype(&::nvmlDeviceGetCount_v2) nvmlDeviceGetCount_v2{nullptr};
    decltype(&::nvmlDeviceGetHandleByIndex_v2) nvmlDeviceGetHandleByIndex_v2{nullptr};
    decltype(&::nvmlDeviceGetMemoryInfo) nvmlDeviceGetMemoryInfo{nullptr};
    decltype(&::nvmlDeviceGetMigMode) nvmlDeviceGetMigMode{nullptr};
    decltype(&::nvmlDeviceGetName) nvmlDeviceGetName{nullptr};
    decltype(&::nvmlDeviceGetPciInfo_v3) nvmlDeviceGetPciInfo_v3{nullptr};
    decltype(&::nvmlDeviceGetPowerManagementLimit) nvmlDeviceGetPowerManagementLimit{nullptr};
    decltype(&::nvmlDeviceGetPowerUsage) nvmlDeviceGetPowerUsage{nullptr};
    decltype(&::nvmlDeviceGetTotalEnergyConsumption) nvmlDeviceGetTotalEnergyConsumption{nullptr};
    decltype(&::nvmlDeviceGetUUID) nvmlDeviceGetUUID{nullptr};
    decltype(&::nvmlErrorString) nvmlErrorString{nullptr};
    decltype(&::nvmlShutdown) nvmlShutdown{nullptr};
    // NOLINTEND(readability-identifier-naming)

  private:
    void* m_nvml_dll{nullptr};

    static std::atomic_int s_dll_counter;
};

// Init the dll counter
std::atomic_int NvmlHandle::s_dll_counter = 0;

struct NvmlState
{
    NvmlState()
    {
        try
        {
            // Try to load the NVML library. If its not found, operate without GPUs
            m_nvml_handle = std::make_unique<NvmlHandle>();
        } catch (std::runtime_error e)
        {
            LOG(WARNING) << "NVML: " << e.what() << ". Setting DeviceCount to 0, CUDA will not be initialized";
            return;
        }

        // Initialize NVML. Must happen first
        auto nvml_status = m_nvml_handle->nvmlInit_v2();

        if (nvml_status != NVML_SUCCESS)
        {
            LOG(WARNING) << "NVML: Error initializing due to '" << m_nvml_handle->nvmlErrorString(nvml_status)
                         << "'. Setting DeviceCount to 0, CUDA will not be initialized";
            return;
        }

        unsigned int visible_devices = 0;

        MRC_CHECK_NVML(m_nvml_handle->nvmlDeviceGetCount_v2(&visible_devices));

        for (decltype(visible_devices) i = 0; i < visible_devices; i++)
        {
            nvmlDevice_t device_handle;
            unsigned int current_mig_mode;
            unsigned int pending_mig_mode;

            auto device_status = m_nvml_handle->nvmlDeviceGetHandleByIndex_v2(i, &device_handle);
            if (device_status != NVML_SUCCESS)
            {
                LOG(WARNING) << "NVML: " << m_nvml_handle->nvmlErrorString(device_status) << "; device with index " << i
                             << " will be ignored";
                continue;
            }

            auto mig_status = m_nvml_handle->nvmlDeviceGetMigMode(device_handle, &current_mig_mode, &pending_mig_mode);
            if (mig_status == NVML_SUCCESS &&
                (current_mig_mode == NVML_DEVICE_MIG_ENABLE || pending_mig_mode == NVML_DEVICE_MIG_ENABLE))
            {
                // let's treat pending as current - MIG mode cannot swap while a cuda process is active, but we may not
                // have initailized CUDA yet, so to avoid any race conditions, we'll error on the side of caution
                if (visible_devices == 1)
                {
                    // if have 1 visible device and it's in MIG mode, then we expect to see only one visible mig
                    // instance
                    m_using_mig = true;

                    // if(number_of_visible_mig_instances == 1) {
                    // if all conditions for running on MIG are met, then we early return from this scope
                    // return;
                    // }

                    LOG(FATAL) << "MRC Issue #205: mig instance queries and enumeration is current not supported";
                }
                else
                {
                    LOG(WARNING) << "NVML visible device #" << i
                                 << " has MIG mode enabled with multiple GPUs visible; this device will be ignored";
                    continue;
                }
            }
            m_accessible_indexes.insert(i);
        }
    }
    ~NvmlState()
    {
        if (m_nvml_handle)
        {
            MRC_CHECK_NVML(m_nvml_handle->nvmlShutdown());
        }
    }

    NvmlHandle& get_handle()
    {
        return *m_nvml_handle;
    }

    const std::set<unsigned int>& accessible_nvml_device_indexes() const
    {
        return m_accessible_indexes;
    }

    bool using_mig() const
    {
        return m_using_mig;
    }

    /**
     * @brief Gets the singleton instance to NvmlState
     *
     * @return NvmlState&
     */
    static NvmlState& instance()
    {
        static NvmlState state;
        return state;
    }

    /**
     * @brief Static method to shorten `NvmlState::instance().get_handle()`
     *
     * @return NvmlHandle&
     */
    static NvmlHandle& handle()
    {
        return NvmlState::instance().get_handle();
    }

  private:
    // this object can also hold the list of device handles that we have access to.
    // - nvmlDeviceGetCount_v2 - will tell us the total number of devices we have access to, i.e. the range of [0, N)
    // device indexes, then enumerate all devices using using nvmlDeviceGetHandleByIndex_v2 and store those handles for
    // which we have permission to access, the call to nvmlDeviceGetHandleByIndex_v2 may return
    // NVML_ERROR_NO_PERMISSION.
    std::set<unsigned int> m_accessible_indexes;

    bool m_using_mig{false};

    std::unique_ptr<NvmlHandle> m_nvml_handle;
};

nvmlDevice_t get_handle_by_id(unsigned int device_id)
{
    nvmlDevice_t handle;
    MRC_CHECK_NVML(NvmlState::handle().nvmlDeviceGetHandleByIndex_v2(device_id, &handle));
    return handle;
}

/*
cpu_set DeviceInfo::Affinity(unsigned int device_id)
{
    nvmlDevice_t gpu       = DeviceInfo::GetHandleById(device_id);
    unsigned long cpu_mask = 0;
    cpu_set cpus;

    CHECK_EQ(nvmlDeviceGetCpuAffinity(gpu, sizeof(cpu_mask), &cpu_mask), NVML_SUCCESS)
        << "Failed to retrieve CpusSet for GPU=" << device_id;

    for (unsigned int i = 0; i < 8 * sizeof(cpu_mask); i++)
    {
        if (test_bit(cpu_mask, i))
        {
            cpus.insert(affinity::system::cpu_from_logical_id(i));
        }
    }

    DLOG(INFO) << "CPU Affinity for GPU " << device_id << ": " << cpus;
    return std::move(cpus);
}
*/

std::size_t DeviceInfo::AccessibleDeviceCount()
{
    return NvmlState::instance().accessible_nvml_device_indexes().size();
}

std::set<unsigned int> DeviceInfo::AccessibleDeviceIndexes()
{
    return NvmlState::instance().accessible_nvml_device_indexes();
}

std::size_t DeviceInfo::Alignment()
{
    struct cudaDeviceProp properties;
    MRC_CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    return properties.textureAlignment;
}

unsigned long long DeviceInfo::DeviceTotalMemory(unsigned int device_id)
{
    nvmlMemory_t info;
    MRC_CHECK_NVML(NvmlState::handle().nvmlDeviceGetMemoryInfo(get_handle_by_id(device_id), &info));
    return info.total;
}

auto DeviceInfo::EnergyConsumption(unsigned int device_id) -> double
{
    unsigned long long energy;
    MRC_CHECK_NVML(NvmlState::handle().nvmlDeviceGetTotalEnergyConsumption(get_handle_by_id(device_id), &energy));
    return static_cast<double>(energy) * 0.001;
}

// This function is taken directly from `hwloc/nvml.h` and updated to use NvmlHandle instead
auto DeviceInfo::GetDeviceCpuset(hwloc_topology_t topology, unsigned int device_id, hwloc_cpuset_t set) -> int
{
/* If we're on Linux, use the sysfs mechanism to get the local cpus */
#define HWLOC_NVML_DEVICE_SYSFS_PATH_MAX 128
    std::array<char, HWLOC_NVML_DEVICE_SYSFS_PATH_MAX> path;
    nvmlReturn_t nvres;
    nvmlPciInfo_t pci;

    if (hwloc_topology_is_thissystem(topology) == 0)
    {
        errno = EINVAL;
        return -1;
    }

    nvres = NvmlState::handle().nvmlDeviceGetPciInfo_v3(get_handle_by_id(device_id), &pci);
    if (NVML_SUCCESS != nvres)
    {
        errno = EINVAL;
        return -1;
    }

    sprintf(path.data(), "/sys/bus/pci/devices/%04x:%02x:%02x.0/local_cpus", pci.domain, pci.bus, pci.device);

    if (hwloc_linux_read_path_as_cpumask(path.data(), set) < 0 || (hwloc_bitmap_iszero(set) != 0))
    {
        hwloc_bitmap_copy(set, hwloc_topology_get_complete_cpuset(topology));
    }

    return 0;
}

std::string DeviceInfo::Name(unsigned int device_id)
{
    std::array<char, 256> buffer;
    MRC_CHECK_NVML(NvmlState::handle().nvmlDeviceGetName(get_handle_by_id(device_id), buffer.data(), 256));
    return buffer.data();
}

std::string DeviceInfo::PCIeBusID(unsigned int device_id)
{
    nvmlPciInfo_t info;
    MRC_CHECK_NVML(NvmlState::handle().nvmlDeviceGetPciInfo_v3(get_handle_by_id(device_id), &info));
    return info.busId;
}

double DeviceInfo::PowerLimit(unsigned int device_id)
{
    unsigned int limit;
    MRC_CHECK_NVML(NvmlState::handle().nvmlDeviceGetPowerManagementLimit(get_handle_by_id(device_id), &limit));
    return static_cast<double>(limit) * 0.001;
}

double DeviceInfo::PowerUsage(unsigned int device_id)
{
    unsigned int power;
    MRC_CHECK_NVML(NvmlState::handle().nvmlDeviceGetPowerUsage(get_handle_by_id(device_id), &power));
    return static_cast<double>(power) * 0.001;
}

std::string DeviceInfo::UUID(unsigned int device_id)
{
    std::array<char, 256> buffer;
    MRC_CHECK_NVML(NvmlState::handle().nvmlDeviceGetUUID(get_handle_by_id(device_id), buffer.data(), 256));
    return buffer.data();
}

}  // namespace mrc::system

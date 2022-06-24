/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "srf/cuda/common.hpp"  // IWYU pragma: associated

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <nvml.h>

#include <array>
#include <cstddef>
#include <memory>
#include <ostream>
#include <set>
#include <string>

#define TEST_BIT(_n, _p) (_n & (1UL << _p))

namespace {
struct NvmlState
{
    NvmlState()
    {
        switch (nvmlInit_v2())
        {
        case NVML_SUCCESS:
            break;
        case NVML_ERROR_DRIVER_NOT_LOADED:
            LOG(WARNING)
                << "NVML: No NVIDIA GPU driver was detected; setting DeviceCount to 0, CUDA will not be initialized";
            return;
        case NVML_ERROR_NO_PERMISSION:
            LOG(WARNING) << "NVML: Access to the NVIDIA GPU driver failed; setting DeviceCount to 0, CUDA will not be "
                            "initialized";
            return;
        default:
            LOG(FATAL) << "NVML_ERROR_UNKNOWN: is a hard fail";
        }

        unsigned int visible_devices = 0;

        CHECK_EQ(nvmlDeviceGetCount_v2(&visible_devices), NVML_SUCCESS);

        for (decltype(visible_devices) i = 0; i < visible_devices; i++)
        {
            nvmlDevice_t device_handle;
            auto device_status = nvmlDeviceGetHandleByIndex_v2(i, &device_handle);
            if (device_status != NVML_SUCCESS)
            {
                LOG(WARNING) << "NVML: " << nvmlErrorString(device_status) << "; device with index " << i
                             << " will be ignored";
                continue;
            }
            m_accessible_indexes.insert(i);
        }
    }
    ~NvmlState()
    {
        CHECK_EQ(nvmlShutdown(), NVML_SUCCESS) << "Failed to Shutdown NVML";
    }

    const std::set<unsigned int>& accessible_nvml_device_indexes() const
    {
        return m_accessible_indexes;
    }

  private:
    // this object can also hold the list of device handles that we have access to.
    // - nvmlDeviceGetCount_v2 - will tell us the total number of devices we have access to, i.e. the range of [0, N)
    // device indexes, then enumerate all devices using using nvmlDeviceGetHandleByIndex_v2 and store those handles for
    // which we have permission to access, the call to nvmlDeviceGetHandleByIndex_v2 may return
    // NVML_ERROR_NO_PERMISSION.
    std::set<unsigned int> m_accessible_indexes;
};

auto nvmlInstatnce = std::make_unique<NvmlState>();

}  // namespace

namespace srf::internal::system {

nvmlDevice_t DeviceInfo::GetHandleById(unsigned int device_id)
{
    nvmlDevice_t handle;
    CHECK_EQ(nvmlDeviceGetHandleByIndex(device_id, &handle), NVML_SUCCESS);
    return handle;
}

/*
cpu_set DeviceInfo::Affinity(int device_id)
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

std::size_t DeviceInfo::Alignment()
{
    struct cudaDeviceProp properties;
    SRF_CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    return properties.textureAlignment;
}

double DeviceInfo::PowerUsage(int device_id)
{
    unsigned int power;
    CHECK_EQ(nvmlDeviceGetPowerUsage(DeviceInfo::GetHandleById(device_id), &power), NVML_SUCCESS);
    return static_cast<double>(power) * 0.001;
}

double DeviceInfo::PowerLimit(int device_id)
{
    unsigned int limit;
    CHECK_EQ(nvmlDeviceGetPowerManagementLimit(DeviceInfo::GetHandleById(device_id), &limit), NVML_SUCCESS);
    return static_cast<double>(limit) * 0.001;
}

std::string DeviceInfo::UUID(int device_id)
{
    std::array<char, 256> buffer;
    CHECK_EQ(nvmlDeviceGetUUID(DeviceInfo::GetHandleById(device_id), buffer.data(), 256), NVML_SUCCESS);
    return buffer.data();
}

std::string DeviceInfo::PCIeBusID(int device_id)
{
    nvmlPciInfo_t info;
    CHECK_EQ(nvmlDeviceGetPciInfo_v3(DeviceInfo::GetHandleById(device_id), &info), NVML_SUCCESS);
    return info.busId;
}

std::size_t DeviceInfo::AccessibleDevices()
{
    CHECK(nvmlInstatnce) << "Failure to Initialize NVML";
    return nvmlInstatnce->accessible_nvml_device_indexes().size();
}

std::set<unsigned int> DeviceInfo::AccessibleDeviceIndexes()
{
    CHECK(nvmlInstatnce) << "Failure to Initialize NVML";
    return nvmlInstatnce->accessible_nvml_device_indexes();
}

nvmlMemory_t DeviceInfo::MemoryInfo(int device_id)
{
    nvmlMemory_t info;
    CHECK_EQ(nvmlDeviceGetMemoryInfo(DeviceInfo::GetHandleById(device_id), &info), NVML_SUCCESS);
    return info;
}

std::string DeviceInfo::Name(int device_id)
{
    std::array<char, 256> buffer;
    CHECK_EQ(nvmlDeviceGetName(DeviceInfo::GetHandleById(device_id), buffer.data(), 256), NVML_SUCCESS);
    return buffer.data();
}

}  // namespace srf::internal::system

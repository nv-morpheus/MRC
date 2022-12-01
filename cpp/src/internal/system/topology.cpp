/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/system/topology.hpp"

#include "internal/system/device_info.hpp"
#include "internal/utils/ranges.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/cuda/common.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/options/topology.hpp"

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <hwloc.h>
#include <hwloc/bitmap.h>
#include <hwloc/nvml.h>
#include <nvml.h>

#include <cstdio>
#include <cstring>
#include <ostream>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

// work-around for known iwyu issue
// https://github.com/include-what-you-use/include-what-you-use/issues/908
// IWYU pragma: no_include <algorithm>
// https://github.com/include-what-you-use/include-what-you-use/issues/166
// IWYU pragma: no_include <ext/alloc_traits.h>

// Topology

namespace mrc::internal::system {

std::shared_ptr<Topology> Topology::Create()
{
    TopologyOptions options;
    return Topology::Create(options);
}

std::shared_ptr<Topology> Topology::Create(const TopologyOptions& options)
{
    hwloc_topology_t system_topology;
    Bitmap cpu_set;

    CHECK_HWLOC(hwloc_topology_init(&system_topology));
    CHECK_HWLOC(hwloc_topology_load(system_topology));

    // use cpu_set of the process (default) or not
    if (options.use_process_cpuset())
    {
        // process cpu_set
        CHECK_HWLOC(hwloc_get_cpubind(system_topology, &cpu_set.bitmap(), HWLOC_CPUBIND_PROCESS));
    }
    else
    {
        // system/machine cpu_set
        cpu_set = Topology::cpuset_for_object(system_topology, HWLOC_OBJ_MACHINE, 0);
    }

    // auto gpu_count                 = DeviceInfo::();
    auto accessible_device_indexes = DeviceInfo::AccessibleDeviceIndexes();
    std::map<int, GpuInfo> gpu_info;  // GpuInfo indexed by CUDA Device ID

    for (const auto& i : accessible_device_indexes)
    {
        GpuInfo info;

        auto* device           = DeviceInfo::GetHandleById(i);
        info.m_name            = DeviceInfo::Name(i);
        info.m_uuid            = DeviceInfo::UUID(i);
        info.m_pcie_bus_id     = DeviceInfo::PCIeBusID(i);
        info.m_memory_capacity = DeviceInfo::MemoryInfo(i).total;
        auto rc                = hwloc_nvml_get_device_cpuset(system_topology, device, &info.m_cpu_set.bitmap());
        CHECK_EQ(rc, 0);

        auto v        = info.cpu_set().vec();
        info.m_cpustr = print_ranges(find_ranges(v));

        // lastly, determine the cuda device id
        auto cuda_rc = cudaDeviceGetByPCIBusId(&info.m_cuda_device_id, info.m_pcie_bus_id.c_str());
        if (cuda_rc != cudaSuccess)
        {
            LOG(WARNING) << "skipping device: " << info.name() << " with pcie: " << info.pcie_bus_id()
                         << "; errmsg=" << __cuda_get_error_string(cuda_rc);
            continue;
        }

        gpu_info[info.cuda_device_id()] = std::move(info);
    }

    return Topology::Create(options, system_topology, cpu_set, std::move(gpu_info));
}

std::shared_ptr<Topology> Topology::Create(const TopologyOptions& options, const protos::Topology& msg)
{
    auto [system_topology, gpus] = Topology::Deserialize(msg);
    CpuSet cpu_set               = Topology::cpuset_for_object(system_topology, HWLOC_OBJ_MACHINE, 0);
    return Topology::Create(options, system_topology, cpu_set, std::move(gpus));
}

std::shared_ptr<Topology> Topology::Create(const TopologyOptions& options,
                                           hwloc_topology_t system_topology,
                                           Bitmap topo_cpu_set,
                                           std::map<int, GpuInfo> gpus)
{
    hwloc_topology_t topology;

    // create a copy that we will use
    CHECK_HWLOC(hwloc_topology_dup(&topology, system_topology));

    // numa policy
    int restrict_numa_flag = 0;
    if (options.restrict_numa_domains())
    {
        // removes numa nodes that do not map to the topo cpu_set
        restrict_numa_flag = HWLOC_RESTRICT_FLAG_REMOVE_CPULESS;
    }

    // we might further restrict the cpu_set based on user passed arguments/envs
    if (!options.user_cpuset().empty())
    {
        auto intersection = topo_cpu_set.set_intersect(options.user_cpuset());
        if (intersection.empty())
        {
            throw exceptions::MrcRuntimeError("intersection between user_cpuset and topo_cpuset is null");
        }
        auto dropped = options.user_cpuset().weight() - intersection.weight();
        if (dropped != 0)
        {
            LOG(WARNING) << "user_cpuset was not a subset of topo_cpuset; " << dropped << " cpus were dropped";
        }
        topo_cpu_set = std::move(intersection);
    }

    // restrict topology to the topo cpu_set
    CHECK_HWLOC(hwloc_topology_restrict(topology, &topo_cpu_set.bitmap(), restrict_numa_flag));
    CHECK_HWLOC(hwloc_topology_refresh(topology));

    auto cpu_set = CpuSet(&topo_cpu_set.bitmap());
    hwloc_topology_destroy(system_topology);

    // collect basic gpu info
    std::map<int, GpuInfo> gpu_info;

    // fiter out any accessible gpus that do not met the requested requirements
    // we can add more filters here - compute capability, memory size, etc.
    for (const auto& [cuda_id, info] : gpus)
    {
        // check gpu's cpu_set against the topo_cpuset, warning if they do not overlap
        auto overlap = info.cpu_set().set_intersect(&topo_cpu_set.bitmap());

        if (options.restrict_gpus() && overlap.empty())
        {
            VLOG(1) << "dropping gpu: " << info
                    << " because restrict_gpus is set to true; fails to overlap with topo cpu_set "
                    << topo_cpu_set.str();
            continue;
        }

        if (options.ignore_dgx_display() and info.name().find("DGX Display") != std::string::npos)
        {
            VLOG(1) << "DGX Display found as an active GPU on the system; dropping this device on request";
            continue;
        }

        gpu_info[cuda_id] = info;
    }

    return std::shared_ptr<Topology>(new Topology(topology, std::move(cpu_set), std::move(gpu_info)));
}

std::shared_ptr<Topology> Topology::Create(const protos::Topology& msg)
{
    auto [topology, gpus] = Topology::Deserialize(msg);
    CpuSet cpu_set(msg.cpu_set());
    return std::shared_ptr<Topology>(new Topology(topology, std::move(cpu_set), std::move(gpus)));
}

std::pair<hwloc_topology_t, std::map<int, GpuInfo>> Topology::Deserialize(const protos::Topology& msg)
{
    hwloc_topology_t topology;
    CHECK_HWLOC(hwloc_topology_init(&topology));
    CHECK_HWLOC(hwloc_topology_set_xmlbuffer(topology, msg.hwloc_xml_string().data(), msg.hwloc_xml_string().size()));
    CHECK_HWLOC(hwloc_topology_load(topology));

    std::map<int, GpuInfo> gpu_info;
    for (const auto& msg : msg.gpu_info())
    {
        auto info                       = GpuInfo::deserialize(msg);
        gpu_info[info.cuda_device_id()] = std::move(info);
    }

    return std::make_pair(topology, std::move(gpu_info));
}

Topology::Topology(hwloc_topology_t topology, CpuSet&& cpu_set, std::map<int, GpuInfo>&& gpu_info) :
  m_topology(topology),
  m_topo_cpuset(cpu_set),
  m_gpu_info(std::move(gpu_info))
{
    CHECK_GT(m_topo_cpuset.weight(), 0);
    VLOG(1) << "topology restricted to cpu_set: " << m_topo_cpuset.str();

    // pull some basic topology info
    m_depth_core = hwloc_get_type_or_below_depth(m_topology, HWLOC_OBJ_CORE);
    m_depth_cpu  = hwloc_get_type_depth(m_topology, HWLOC_OBJ_PU);
    m_depth_numa = hwloc_get_type_depth(m_topology, HWLOC_OBJ_NUMANODE);
    DCHECK_EQ(m_depth_numa, HWLOC_TYPE_DEPTH_NUMANODE);  // if this passes, no need to call hwloc_get_type_depth

    // collect numa_node info
    for (int i = 0; i < numa_count(); i++)
    {
        auto* obj = hwloc_get_obj_by_type(m_topology, HWLOC_OBJ_NUMANODE, i);
        m_numa_cpusets.emplace_back(obj->cpuset);
    }
}

Topology::~Topology()
{
    if (m_topology != nullptr)
    {
        hwloc_topology_destroy(m_topology);
    }
}

int Topology::depth_for_object(hwloc_obj_type_t obj) const
{
    int rc = -1;
    switch (obj)
    {
    case HWLOC_OBJ_MACHINE:
    case HWLOC_OBJ_NUMANODE:
    case HWLOC_OBJ_PU:
        rc = hwloc_get_type_depth(m_topology, obj);
        break;
    case HWLOC_OBJ_CORE:
        rc = hwloc_get_type_or_below_depth(m_topology, HWLOC_OBJ_CORE);
        break;
    default:
        LOG(FATAL) << "obj with value: " << obj << " not support by this method";
    }
    CHECK_NE(rc, -1);
    return rc;
}

hwloc_obj_t Topology::object_at_depth(int depth, int id) const
{
    return object_at_depth(m_topology, depth, id);
}
hwloc_obj_t Topology::object_at_depth(hwloc_topology_t topo, int depth, int id)
{
    auto dt = hwloc_get_depth_type(topo, depth);
    CHECK_NE(dt, ((hwloc_obj_type_t)-1)) << "invalid depth";

    auto count = hwloc_get_nbobjs_by_depth(topo, depth);
    CHECK_LT(id, count) << "invalid id";

    return hwloc_get_obj_by_depth(topo, depth, id);
}
CpuSet Topology::cpuset_for_object(int depth, int id) const
{
    return cpuset_for_object(m_topology, depth, id);
}
CpuSet Topology::cpuset_for_object(hwloc_topology_t topo, int depth, int id)
{
    auto* obj = object_at_depth(topo, depth, id);
    return CpuSet(obj->cpuset);
}
NumaSet Topology::numaset_for_object(int depth, int id) const
{
    auto* obj = object_at_depth(m_topology, depth, id);
    return NumaSet(obj->nodeset);
}
const CpuSet& Topology::cpu_set() const
{
    return m_topo_cpuset;
}
const CpuSet& Topology::numa_cpuset(std::uint32_t id) const
{
    CHECK_LT(id, m_numa_cpusets.size());
    return m_numa_cpusets.at(id);
}

std::uint32_t Topology::object_count_at_depth(int depth) const
{
    return hwloc_get_nbobjs_by_depth(m_topology, depth);
}
std::uint32_t Topology::core_count() const
{
    return object_count_at_depth(m_depth_core);
}
std::uint32_t Topology::cpu_count() const
{
    return object_count_at_depth(m_depth_cpu);
}
std::uint32_t Topology::numa_count() const
{
    return object_count_at_depth(m_depth_numa);
}

NumaSet Topology::numaset_for_cpuset(const CpuSet& cpu_set) const
{
    NumaSet numa_set;
    CHECK_HWLOC(hwloc_cpuset_to_nodeset(handle(), &cpu_set.bitmap(), &numa_set.bitmap()));
    return numa_set;
}

std::uint32_t Topology::gpu_count() const
{
    return m_gpu_info.size();
}
hwloc_topology_t Topology::handle() const
{
    return m_topology;
}
Bitmap& Topology::topo_bitset()
{
    return m_topo_cpuset;
}
std::string Topology::export_xml() const
{
    char* buffer = nullptr;
    int length   = 0;

    CHECK_HWLOC(hwloc_topology_export_xmlbuffer(m_topology, &buffer, &length, 0));

    std::string xml;
    xml.resize(length);
    std::memcpy(xml.data(), buffer, length);

    return xml;
}
protos::Topology Topology::serialize() const
{
    protos::Topology msg;
    msg.set_hwloc_xml_string(export_xml());
    msg.set_cpu_set(m_topo_cpuset.str());
    for (const auto& [id, info] : m_gpu_info)
    {
        auto* i = msg.add_gpu_info();
        *i      = info.serialize();
    }
    return msg;
}

const std::map<int, GpuInfo>& Topology::gpu_info() const
{
    return m_gpu_info;
}

void Topology::serialize_to_file(std::string path)
{
    auto* file       = fopen(path.c_str(), "wb");
    int posix_handle = fileno(file);
    auto topology    = system::Topology::Create();
    auto system      = topology->serialize().SerializePartialToFileDescriptor(posix_handle);
    fclose(file);
}

protos::Topology Topology::deserialize_from_file(std::string path)
{
    protos::Topology msg;
    auto* file       = fopen(path.c_str(), "rb");
    int posix_handle = fileno(file);
    CHECK(msg.ParseFromFileDescriptor(posix_handle));
    fclose(file);
    return msg;
}

bool Topology::contains(const CpuSet& cpu_set) const
{
    return bool(hwloc_bitmap_isincluded(&cpu_set.bitmap(), &this->cpu_set().bitmap()));
}
}  // namespace mrc::internal::system

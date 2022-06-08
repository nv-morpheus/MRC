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

#include "internal/system/partitions.hpp"

#include "internal/system/system.hpp"
#include "internal/system/topology.hpp"
#include "internal/utils/shared_resource_bit_map.hpp"
#include "srf/options/placement.hpp"
#include "srf/utils/bytes_to_string.hpp"

#include <glog/logging.h>
#include <hwloc.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

static void div_even(std::int32_t n, std::int32_t np, std::int32_t me, std::int32_t& nr, std::int32_t& sr)
{
    nr      = n / np;
    auto ne = n % np;

    if (me < ne)
    {
        // nr++;
        sr = (nr + 1) * me;
    }
    else
    {
        sr = (nr + 1) * ne + nr * (me - ne);
    }
}

namespace srf::internal::system {

Partitions::Partitions(const System& system) : Partitions(system.topology(), system.options()) {}

Partitions::Partitions(const Topology& topology, const Options& options)
{
    VLOG(10) << "forming memory and device partitions";

    // the number of host partitions is determined by the placement options

    SharedResourceBitMap gpus_per_numa_node;

    // for each gpu in topology determine which numa node the gpu belongs
    // the number of entries in the SharedResourcesBitMap denotes the number of NUMA nodes that have at least one device
    for (const auto& [gpu_id, info] : topology.gpu_info())
    {
        NumaSet node_set;
        auto rc = hwloc_cpuset_to_nodeset(topology.handle(), &info.cpu_set().bitmap(), &node_set.bitmap());
        CHECK_NE(rc, -1);
        if (node_set.weight() != 0)
        {
            CHECK_EQ(node_set.weight(), 1);
            gpus_per_numa_node.insert(node_set, gpu_id);
        }
    }

    m_cpu_strategy = options.placement().cpu_strategy();

    if (options.placement().cpu_strategy() == PlacementStrategy::PerNumaNode && !gpus_per_numa_node.map().empty())
    {
        bool symmetric = true;
        int div        = topology.gpu_count() / gpus_per_numa_node.map().size();
        int rem        = topology.gpu_count() % gpus_per_numa_node.map().size();

        if (topology.numa_count() != gpus_per_numa_node.map().size())
        {
            symmetric = false;
        }

        if (rem != 0)
        {
            symmetric = false;
        }

        for (const auto [numa_id, gpu_bitmap] : gpus_per_numa_node.map())
        {
            if (gpu_bitmap.weight() != div)
            {
                symmetric = false;
            }
        }

        if (!symmetric)
        {
            VLOG(1) << "warning: asymmetric topology detected; PerNumaNode strategy cannot be used; falling back to "
                       "PerMachine";
            m_cpu_strategy = PlacementStrategy::PerMachine;
        }
    }

    m_device_to_host_strategy = options.placement().resources_strategy();
    if (topology.cpu_count() < topology.gpu_count() && m_device_to_host_strategy == PlacementResources::Dedicated)
    {
        VLOG(1) << "warning: cpu_count=" << topology.cpu_count()
                << " is less than the number of cuda devices; unable to allocate dedicated cpu cores per device; "
                   "falling back to shared host partition";
        m_device_to_host_strategy = PlacementResources::Shared;
    }

    hwloc_obj_type_t host_partition_over_obj = HWLOC_OBJ_MACHINE;
    if (m_cpu_strategy == PlacementStrategy::PerNumaNode)
    {
        host_partition_over_obj = HWLOC_OBJ_NUMANODE;
    }

    auto partition_depth   = topology.depth_for_object(host_partition_over_obj);
    auto partition_size    = topology.object_count_at_depth(partition_depth);
    int gpus_per_partition = topology.gpu_count() / partition_size;
    CHECK_EQ(topology.gpu_count() % partition_size, 0);

    VLOG(10) << "detected " << topology.gpu_count() << " nvidia gpus; " << partition_size << " host cpu/memory domains";

    // GPU selection by cpu_set/numa_set with max overlap
    std::map<int, GpuInfo> remaining_gpus(topology.gpu_info());

    // for each remaining gpu, score the gpu's cpu_set against the host_partition's cpu_set, pick the gpu with the
    // highest overlap and use the lowest indexed gpu in case of a tie
    auto device_best_match = [&remaining_gpus, &topology](const CpuSet& cpu_set) mutable {
        int top_idx    = -1;
        long top_score = -1;

        for (const auto& [gpu_id, info] : remaining_gpus)
        {
            auto intersection = cpu_set.set_intersect(info.cpu_set());
            if (intersection.weight() > top_score)
            {
                top_idx = gpu_id;
            }
        }
        remaining_gpus.erase(top_idx);
        return top_idx;
    };

    for (int p_id = 0; p_id < partition_size; p_id++)
    {
        auto* obj  = topology.object_at_depth(partition_depth, p_id);
        auto count = std::max(gpus_per_partition, 1);

        std::vector<CpuSet> cpu_sets;
        std::size_t partition_memory = obj->total_memory;

        CpuSet cpu_set(obj->cpuset);
        VLOG(10) << "host cpu/memory domain: " << p_id << " has " << cpu_set.weight() << " logical cpus ("
                 << cpu_set.str() << ") with " << bytes_to_string(partition_memory) << " memory";

        if (m_device_to_host_strategy == PlacementResources::Dedicated)
        {
            auto* root      = topology.object_at_depth(partition_depth, p_id);
            auto core_count = hwloc_get_nbobjs_inside_cpuset_by_type(topology.handle(), root->cpuset, HWLOC_OBJ_CORE);
            CHECK_NE(core_count, -1);
            CHECK_LE(count, core_count);
            VLOG(10) << "host resources will be split into " << count << " partitions across " << core_count
                     << " cores";

            for (int i = 0; i < count; i++)
            {
                std::int32_t nc;
                std::int32_t sc;
                CpuSet cpu_set;
                div_even(core_count, count, i, nc, sc);

                DVLOG(20) << "split " << i << ": start_core: " << sc << "; ncores: " << nc;

                for (int ic = sc; ic < sc + nc; ic++)
                {
                    auto* core_obj =
                        hwloc_get_obj_inside_cpuset_by_type(topology.handle(), root->cpuset, HWLOC_OBJ_CORE, ic);
                    CpuSet core_cpuset(core_obj->cpuset);
                    DVLOG(30) << "ic: " << ic << "; cpuset: " << core_cpuset.str();
                    cpu_set.append(CpuSet(core_obj->cpuset));
                }

                cpu_sets.push_back(std::move(cpu_set));
                DVLOG(20) << "split cpuset: " << cpu_sets.back();
            }
            partition_memory /= count;
        }
        else
        {
            // if shared, we will only push back 1 HostPartition per object at the given depth
            // i.e. 1 total if PerMachine, numa_count if PerNumaNode
            VLOG(10) << "host resources will not be further subdivided";
            HostPartition host_partition(topology.cpuset_for_object(partition_depth, p_id),
                                         topology.numaset_for_object(partition_depth, p_id),
                                         partition_memory);
            m_host_partitions.push_back(std::move(host_partition));

            VLOG(10) << "host_partition_id: " << m_host_partitions.size() - 1 << " contains "
                     << m_host_partitions.back().cpu_set().weight() << " logical cpus ("
                     << m_host_partitions.back().cpu_set().str() << ") with " << bytes_to_string(partition_memory)
                     << " memory";
        }

        for (int s_id = 0; s_id < count; s_id++)
        {
            if (m_device_to_host_strategy == PlacementResources::Dedicated)
            {
                // if dedicated, we push back 1 HostPartition for each dedicated partition
                CHECK_LT(s_id, cpu_sets.size());
                HostPartition host_partition(
                    cpu_sets[s_id], topology.numaset_for_cpuset(cpu_sets[s_id]), partition_memory);
                m_host_partitions.push_back(std::move(host_partition));

                VLOG(10) << "host_partition_id: " << m_host_partitions.size() - 1 << " contains "
                         << m_host_partitions.back().cpu_set().weight() << " logical cpus ("
                         << m_host_partitions.back().cpu_set().str() << ") with " << bytes_to_string(partition_memory)
                         << " memory";
            }

            CHECK_GT(m_host_partitions.size(), 0);
            auto host_partition_id = m_host_partitions.size() - 1;

            if (!remaining_gpus.empty())
            {
                auto cuda_id = device_best_match(m_host_partitions[host_partition_id].cpu_set());

                auto device_partition_id     = m_device_partitions.size();
                const auto& device_partition = m_device_partitions.emplace_back(
                    topology.gpu_info().at(cuda_id), m_host_partitions, host_partition_id);
                m_host_partitions[host_partition_id].add_device_partition_id(device_partition_id);

                VLOG(10) << "assigning cuda_device_id: " << device_partition.cuda_device_id()
                         << "; pcie: " << device_partition.pcie_bus_id()
                         << " to host_partition_id: " << host_partition_id;
            }
        }
    }

    CHECK_EQ(topology.gpu_count(), m_device_partitions.size());
    CHECK_EQ(remaining_gpus.size(), 0);

    // generate engine factory cpu sets

    // eventually we will pass the public_api::HostPartition to the ResourceBuilder

    for (auto& partition : m_host_partitions)
    {
        VLOG(10) << "evaluating engine factory cpu sets for host_partition " << partition.cpu_set().str();
        partition.set_engine_factory_cpu_sets(options);
    }
}

const std::vector<HostPartition>& Partitions::host_partitions() const
{
    return m_host_partitions;
}

const std::vector<DevicePartition>& Partitions::device_partitions() const
{
    return m_device_partitions;
}

const PlacementStrategy& Partitions::cpu_strategy() const
{
    return m_cpu_strategy;
}

const PlacementResources& Partitions::device_to_host_strategy() const
{
    return m_device_to_host_strategy;
}

const HostPartition& Partitions::host_partition_containing(const CpuSet& cpu_set) const
{
    CHECK(cpu_set.weight() > 0);
    for (const auto& partition : m_host_partitions)
    {
        if (partition.cpu_set().contains(cpu_set))
        {
            return partition;
        }
    }
    LOG(FATAL) << "unable to determine host_partition which contains cpu_set: " << cpu_set.str();
}

}  // namespace srf::internal::system

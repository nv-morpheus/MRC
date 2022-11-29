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

#include "internal/system/host_partition.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/options/options.hpp"

#include <utility>

namespace mrc::internal::system {

HostPartition::HostPartition(CpuSet cpu_set, NumaSet numa_set, std::size_t total_memory) :
  m_cpu_set(std::move(cpu_set)),
  m_numa_set(std::move(numa_set)),
  m_total_memory(total_memory)
{}

const CpuSet& HostPartition::cpu_set() const
{
    return m_cpu_set;
}
const NumaSet& HostPartition::numa_set() const
{
    return m_numa_set;
}
std::size_t HostPartition::host_memory_capacity() const
{
    return m_total_memory;
}
const std::vector<int>& HostPartition::device_partition_ids() const
{
    return m_device_partitions;
}

void HostPartition::add_device_partition_id(int gpu_id)
{
    m_device_partitions.push_back(gpu_id);
}

void HostPartition::set_engine_factory_cpu_sets(const Topology& topology, const Options& options)
{
    m_engine_factory_cpu_sets = generate_engine_factory_cpu_sets(topology, options, cpu_set());
}

const EngineFactoryCpuSets& HostPartition::engine_factory_cpu_sets() const
{
    return m_engine_factory_cpu_sets;
}
}  // namespace mrc::internal::system

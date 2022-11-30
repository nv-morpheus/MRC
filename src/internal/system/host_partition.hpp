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

#pragma once

#include "internal/system/engine_factory_cpu_sets.hpp"
#include "internal/system/topology.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/options/options.hpp"

#include <cstddef>
#include <vector>

namespace mrc {
class Options;  // IWYU pragma: keep
}

namespace mrc::internal::system {

class HostPartition
{
  public:
    HostPartition(CpuSet cpu_set, NumaSet numa_set, std::size_t total_memory);
    virtual ~HostPartition() = default;

    const CpuSet& cpu_set() const;
    const NumaSet& numa_set() const;
    std::size_t host_memory_capacity() const;
    const std::vector<int>& device_partition_ids() const;

    void add_device_partition_id(int gpu_id);
    void set_engine_factory_cpu_sets(const Topology& topology, const Options& options);

    const EngineFactoryCpuSets& engine_factory_cpu_sets() const;

  private:
    CpuSet m_cpu_set;
    NumaSet m_numa_set;
    std::size_t m_total_memory;
    std::vector<int> m_device_partitions;
    EngineFactoryCpuSets m_engine_factory_cpu_sets;
};

}  // namespace mrc::internal::system

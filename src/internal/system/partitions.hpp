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

#include "internal/system/device_partition.hpp"
#include "internal/system/gpu_info.hpp"
#include "internal/system/host_partition.hpp"

#include <srf/core/bitmap.hpp>
#include <srf/options/options.hpp>
#include <srf/options/placement.hpp>

#include <vector>

namespace srf::internal::system {

class System;
class Topology;  // IWYU pragma: keep

class Partitions
{
  public:
    Partitions(const Topology& topology, const Options& options);
    Partitions(const System& system);

    const std::vector<HostPartition>& host_partitions() const;
    const std::vector<DevicePartition>& device_partitions() const;

    const HostPartition& host_partition_containing(const CpuSet& cpu_set) const;

    const PlacementStrategy& cpu_strategy() const;
    const PlacementResources& device_to_host_strategy() const;

  private:
    std::vector<HostPartition> m_host_partitions;
    std::vector<DevicePartition> m_device_partitions;
    PlacementStrategy m_cpu_strategy;
    PlacementResources m_device_to_host_strategy;
};

}  // namespace srf::internal::system

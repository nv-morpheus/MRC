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
#include "internal/system/host_partition.hpp"
#include "internal/system/partition.hpp"
#include "internal/system/topology.hpp"  // todo(iwyu) - forward declare Topology?

#include "mrc/options/options.hpp"
#include "mrc/options/placement.hpp"

#include <vector>

namespace mrc::internal::system {

class System;

class Partitions
{
  public:
    Partitions(const Topology& topology, const Options& options);
    Partitions(const System& system);

    // The host and device partitions are hierarchical where there is a possibility, depending on options provided,
    // where more than one cuda device shares the same host partition, so those host resources are shared.
    //
    // However, each device defines an entry in the flattened partition list since it will have its own unique network
    // and device resources.
    //
    // We flatten the partitions such that it is an ordered list where the first N partitions correspond to N cuda
    // devices, sorted by cuda_device_id. If there are more than N partition, those partitions do not have devices
    // attached.
    //
    // The flattened partitions provide the view of what's available to a given GPU.

    const std::vector<HostPartition>& host_partitions() const;
    const std::vector<DevicePartition>& device_partitions() const;
    const std::vector<Partition>& flattened() const;

    const PlacementStrategy& cpu_strategy() const;
    const PlacementResources& device_to_host_strategy() const;

  private:
    std::vector<Partition> m_partitions;
    std::vector<HostPartition> m_host_partitions;
    std::vector<DevicePartition> m_device_partitions;

    PlacementStrategy m_cpu_strategy;
    PlacementResources m_device_to_host_strategy;
};

}  // namespace mrc::internal::system

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

// todo(ryan) - rename to PartitionStrategy

/**
 * Placement Options
 *
 * The selection of placement options effects on how the physical node and its resources are divided up into parition
 * groups. There are two categories, PlacementStrategy and PlacementResources.
 *
 * PlacementStrategy determines if we treating the machine/node as a single collection of resources or a collection of
 * resources assigned to specific numa domains.
 *
 * PlacementResources determines if the collection of GPUs to assocated with the cpu/host_mem resources of a
 * PlacementStrategy are shared or split evenly with respect to the number of GPUs associated within a
 * PlacementStrategy.
 *
 * Note, if PlacementStrategy::PerNumaNode is selected, then the number of GPUs per NUMA node must be symmetric;
 * otherwise, the PlacementStrategy will fallback to PerMachine and a warning provided.
 *
 * Examples:
 *
 * Machine1 has 4 NUMA nodes with each with 16cores/32cpus, 128GB of host memory, and one NVIDIA GPU for a machine total
 * of 64cores/128cpus, 512 GB of system memory, and 4 NVIDIA GPUs.
 *
 * Scenario 1:
 * - the process has access to the entire machine with the default affinity set to all cpus, all GPUs are visible
 * - PlacementStrategy == PerMachine
 * - PlacementResources == Dedicated
 * ==> 4 GpuPartitions are created each with a dedicated HostPartition of 32 unique cpus_ids and 128 GB of host memory
 * ==> the gpus in each host partition not by numa aligned as the order in which the subdivision of host resources is
 * split does not take into consideration numa placement, simply enumerating all the devices, spliting the resources,
 * and making assignments in the order the devices were enumerated
 * ==> todo(create_issue) - an effort should be made to align gpus and paritions similar to the next scenario, but no
 * guarantees
 *
 * Scenario 2 (optimal for DGX)
 * - the process has access to the entire machine with the default affinity set to all cpus, all GPUs are visible
 * - PlacementStrategy == PerNumaNode
 * - PlacementResources == Dedicated
 * ==> 4 GpuPartitions are created each with a dedicated HostPartition of 32 unique cpus_ids and 128 GB of host memory
 * ==> the gpus in each host partition are properly aligned with the cpu cores and host memory that are "closest"
 *
 * Scenario 3:
 * - the process has access to the entire machine with the default affinity set to all cpus, 2/4 GPUs are visible to the
 * process
 * - PlacementStrategy == PerNumaNode
 * - PlacementResources == Dedicated
 * ==> PerNumaNode warns of asymmetry; falls back to PerMachine
 * ==> 2 GpuParitions are created each a HostPartition with 64cpus and 256GB memory
 * ==> the gpus may not be optimally aligned
 * ==> todo(create_issue) - an effort should be made to align gpus and paritions similar to the previous scenario, but
 * no guarantees
 *
 * Scenario 4
 * - the process has access to the entire machine with the default affinity set to all cpus, all GPUs are visible
 * - PlacementStrategy == PerMachine
 * - PlacementResources == Shared
 * ==> 4 GpuPartitions are created each with a single HostPartition visible to each DevicePartition
 * ==> The HostPartition sees all cpus and memory.
 *
 * Scenario 5
 * - the process has access to the entire machine with the default affinity set to all cpus, all GPUs are visible
 * - PlacementStrategy == PerNumaNode
 * - PlacementResources == Shared
 * ==> Equivalent to Scenario #2
 * ==> Behavior of this Scenario would differ from #2 if the number of GPUs assigned to each NUMA node was greater
 *     than 1.
 * ==> If the number of GPUs per NUMA node was 2 (8 for the machine), then #2 would have 8 HostParitions where this
 *     Scenario would have 4 HostPartition.
 */

/**
 * @brief Strategy for partitioning the machine into groups
 */
namespace mrc {
enum class PlacementStrategy
{
    PerMachine,
    PerSocket,
    PerNumaNode,
};

/**
 * @brief Strategy for
 */
enum class PlacementResources
{
    Dedicated,
    Shared,
};

class PlacementOptions
{
  public:
    PlacementOptions() = default;

    PlacementOptions& cpu_strategy(const PlacementStrategy& strategy);
    PlacementOptions& resources_strategy(const PlacementResources& strategy);

    [[nodiscard]] PlacementStrategy cpu_strategy() const;
    [[nodiscard]] PlacementResources resources_strategy() const;

  private:
    PlacementStrategy m_cpu_strategy{PlacementStrategy::PerMachine};
    PlacementResources m_resources_strategy{PlacementResources::Shared};
};

}  // namespace mrc

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

#include "internal/resources/system_resources.hpp"
#include "internal/system/system.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace srf::internal::resources {

class PartitionResources;

/**
 * @brief Constructs a flat list of System resources.
 *
 * SystemResources is not a flat list. Multiple GPU could be assigned to the same HostPartiton. This class,
 * ResourcePartitions, flattens SystemResources into an ordered list of Resources. If N GPUs are present on the
 * system, then the partitions corresponding to [0,N) contain devices sorted by CUDA Device ID; paritions without
 * devices would follow from [N, M) where M is the total number of partitions.
 *
 *
 *
 */
class ResourcePartitions : public SystemResources
{
  public:
    ResourcePartitions(std::shared_ptr<system::System> system);

    std::size_t gpu_count() const;
    std::size_t partitions() const;

    PartitionResources& partition(std::size_t partition_id) const;
    std::shared_ptr<PartitionResources> partition_shared(std::size_t partition_id) const;

  private:
    std::vector<std::shared_ptr<PartitionResources>> m_resources;
};

inline std::shared_ptr<ResourcePartitions> make_resource_partitions(std::shared_ptr<system::System> system)
{
    return std::make_shared<ResourcePartitions>(std::move(system));
}

}  // namespace srf::internal::resources

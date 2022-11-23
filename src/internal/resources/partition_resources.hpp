/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/memory/device_resources.hpp"
#include "internal/memory/host_resources.hpp"
#include "internal/network/resources.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/runnable/resources.hpp"

#include <cstddef>
#include <optional>

namespace mrc::internal::resources {

/**
 * @brief Partition Resources define the set of Resources available to a given Partition
 *
 * This class does not own the actual resources, that honor is bestowed on the resources::Manager. This class is
 * constructed and owned by the resources::Manager to ensure validity of the references.
 */
class PartitionResources final : public PartitionResourceBase
{
  public:
    PartitionResources(runnable::Resources& runnable_resources,
                       std::size_t partition_id,
                       memory::HostResources& host,
                       std::optional<memory::DeviceResources>& device,
                       std::optional<network::Resources>& network);

    memory::HostResources& host();
    std::optional<memory::DeviceResources>& device();
    std::optional<network::Resources>& network();

  private:
    memory::HostResources& m_host;
    std::optional<memory::DeviceResources>& m_device;
    std::optional<network::Resources>& m_network;
};

}  // namespace mrc::internal::resources

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

#include "internal/resources/device_resources.hpp"
#include "internal/resources/host_resources.hpp"

#include <cstdint>
#include <memory>

namespace srf::internal::resources {

/**
 * @brief Set of Resources assigned to a specific parition.
 *
 * PartitionResources constructs a flat list of Resources from SystemResources.
 */
class PartitionResources
{
  public:
    PartitionResources(std::uint32_t partition_id,
                       std::shared_ptr<HostResources> host,
                       std::shared_ptr<DeviceResources> device);

    const std::uint32_t& partition_id() const;

    HostResources& host() const;
    DeviceResources& device() const;

  private:
    std::uint32_t m_partition_id;
    std::shared_ptr<HostResources> m_host_resources;
    std::shared_ptr<DeviceResources> m_device_resources;
};

}  // namespace srf::internal::resources

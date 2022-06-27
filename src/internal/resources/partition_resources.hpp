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

#include "internal/memory/resources.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/ucx/resources.hpp"

#include "srf/memory/resources/memory_resource.hpp"

#include <glog/logging.h>

#include <optional>

namespace srf::internal::resources {

class HostResources;
class DeviceResources;

/**
 * @brief Owner of most resources assigned to a specific partition.
 */
class PartitionResources final : public PartitionResourceBase
{
  public:
    PartitionResources(runnable::Resources& runnable_resources,
                       std::size_t partition_id,
                       HostResources& host,
                       std::optional<ucx::Resources>& ucx,
                       std::optional<DeviceResources>& device);

    HostResources& host();
    std::optional<ucx::Resources>& ucx();
    std::optional<DeviceResources>& device();

  private:
    HostResources& m_host;
    std::optional<ucx::Resources>& m_ucx;
    std::optional<DeviceResources>& m_device;
};

}  // namespace srf::internal::resources

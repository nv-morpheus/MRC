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

#include "internal/resources/host_resources.hpp"
#include "internal/system/device_partition.hpp"

#include <memory>

namespace srf::internal::resources {

class DeviceResources
{
  public:
    DeviceResources(const system::DevicePartition& partition, std::shared_ptr<HostResources>);

    int cuda_device_id() const;
    const system::DevicePartition& partition() const;

    HostResources& host() const;
    std::shared_ptr<HostResources> host_shared() const;

  private:
    const system::DevicePartition& m_partition;
    std::shared_ptr<HostResources> m_host_resources;

    // todo(update) - with cuda::memory_resource when ready
    // rmm::device_memory_resource
};

}  // namespace srf::internal::resources

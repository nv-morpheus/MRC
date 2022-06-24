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

#include "internal/resources/partition_resources.hpp"

#include "srf/exceptions/runtime_error.hpp"

#include <glog/logging.h>

#include <utility>

namespace srf::internal::resources {

PartitionResources::PartitionResources(std::uint32_t partition_id,
                                       std::shared_ptr<HostResources> host,
                                       std::shared_ptr<DeviceResources> device) :
  m_partition_id(partition_id),
  m_host_resources(std::move(host)),
  m_device_resources(std::move(device))
{}

HostResources& PartitionResources::host() const
{
    if (m_device_resources)
    {
        return m_device_resources->host();
    }
    CHECK(m_host_resources);
    return *m_host_resources;
}

DeviceResources& PartitionResources::device() const
{
    if (m_device_resources)
    {
        return *m_device_resources;
    }
    throw exceptions::SrfRuntimeError("no device associated with this partition");
}
const std::uint32_t& PartitionResources::partition_id() const
{
    return m_partition_id;
};

}  // namespace srf::internal::resources

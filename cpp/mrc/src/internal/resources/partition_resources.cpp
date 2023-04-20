/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <optional>

namespace mrc::internal::resources {

PartitionResources::PartitionResources(runnable::RunnableResources& runnable_resources,
                                       std::size_t partition_id,
                                       memory::HostResources& host,
                                       std::optional<memory::DeviceResources>& device,
                                       std::optional<ucx::UcxResources>& ucx) :
  PartitionResourceBase(runnable_resources, partition_id),
  m_host(host),
  m_device(device),
  m_ucx(ucx)
{}

memory::HostResources& PartitionResources::host()
{
    return m_host;
}

std::optional<memory::DeviceResources>& PartitionResources::device()
{
    return m_device;
}

std::optional<network::NetworkResources>& PartitionResources::network()
{
    return m_network;
}

std::optional<ucx::UcxResources>& PartitionResources::ucx() const
{
    return m_ucx;
}

}  // namespace mrc::internal::resources

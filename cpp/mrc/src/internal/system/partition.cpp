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

#include "internal/system/partition.hpp"

#include "mrc/exceptions/runtime_error.hpp"

#include <glog/logging.h>

#include <ostream>
#include <utility>

namespace mrc::internal::system {

const HostPartition& Partition::host() const
{
    CHECK(m_host);
    return *m_host;
}

const DevicePartition& Partition::device() const
{
    if (m_device)
    {
        return *m_device;
    }
    LOG(ERROR) << "attemping to access an unassigned DevicePartition";
    throw exceptions::MrcRuntimeError("no device partition available");
}
bool Partition::has_device() const
{
    return (static_cast<bool>(m_device));
}
Partition::Partition(std::size_t host_partition_id,
                     std::shared_ptr<const HostPartition> host,
                     std::shared_ptr<const DevicePartition> device) :
  m_host_partition_id(host_partition_id),
  m_host(std::move(host)),
  m_device(std::move(device))
{}
size_t Partition::host_partition_id() const
{
    return m_host_partition_id;
}
}  // namespace mrc::internal::system

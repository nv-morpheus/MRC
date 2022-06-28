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

#include "internal/system/host_partition_provider.hpp"

#include "internal/system/partitions.hpp"
#include "internal/system/system.hpp"

#include <glog/logging.h>

#include <vector>

namespace srf::internal::system {

HostPartitionProvider::HostPartitionProvider(const SystemProvider& _system, std::size_t _host_partition_id) :
  SystemProvider(_system),
  m_host_partition_id(_host_partition_id)
{
    CHECK_LT(m_host_partition_id, this->system().partitions().host_partitions().size());
}

std::size_t HostPartitionProvider::host_partition_id() const
{
    return m_host_partition_id;
}

const HostPartition& HostPartitionProvider::host_partition() const
{
    CHECK_LT(m_host_partition_id, system().partitions().host_partitions().size());
    return system().partitions().host_partitions().at(m_host_partition_id);
}
}  // namespace srf::internal::system

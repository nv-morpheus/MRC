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

#include "internal/system/partition_provider.hpp"

#include "internal/system/partitions.hpp"
#include "internal/system/system.hpp"

#include <glog/logging.h>

#include <vector>

namespace mrc::internal::system {

PartitionProvider::PartitionProvider(SystemProvider& system, std::size_t partition_id) :
  SystemProvider(system),
  m_partition_id(partition_id)
{
    CHECK_LT(m_partition_id, this->system().partitions().flattened().size());
}

std::size_t PartitionProvider::partition_id() const
{
    return m_partition_id;
}
const Partition& PartitionProvider::partition() const
{
    return system().partitions().flattened().at(m_partition_id);
}
}  // namespace mrc::internal::system

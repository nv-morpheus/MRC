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

#include "internal/system/host_partition.hpp"
#include "internal/system/system_provider.hpp"

#include <cstddef>

namespace mrc::internal::system {

/**
 * @brief Extends SystemProvider to add host_partition_id and host_partition info.
 *
 * This is a common base class for resources that are tied to HostPartitions.
 */
class HostPartitionProvider : public SystemProvider
{
  public:
    HostPartitionProvider(const SystemProvider& _system, std::size_t _host_partition_id);

    std::size_t host_partition_id() const;
    const HostPartition& host_partition() const;

  private:
    const std::size_t m_host_partition_id;
};

}  // namespace mrc::internal::system

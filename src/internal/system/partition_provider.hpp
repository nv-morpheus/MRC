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

#include "internal/system/partition.hpp"
#include "internal/system/system_provider.hpp"

#include <cstddef>

namespace srf::internal::system {

/**
 * @brief Extends SystemProvider to provide access to partition_id and partition details.
 *
 * This is a common base classes to resources that are tied to the flattened partitions list, e.g. cuda devices memory
 * resources.
 *
 * @note See Partitions for a more detailed description for the information contained.
 */
class PartitionProvider : public SystemProvider
{
  public:
    PartitionProvider(SystemProvider& system, std::size_t partition_id);

    std::size_t partition_id() const;
    const Partition& partition() const;

  private:
    std::size_t m_partition_id;
};

}  // namespace srf::internal::system

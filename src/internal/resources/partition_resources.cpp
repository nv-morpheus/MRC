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

#include "internal/resources/partition_resources.hpp"

namespace srf::internal::resources {

PartitionResources::PartitionResources(runnable::Resources& runnable_resources, std::size_t partition_id) :
  RunnableProvider(runnable_resources, partition_id)
{
    if (!system().options().architect_url().empty())
    {
        m_network.emplace(*this);
    }
}

std::optional<network::Resources>& PartitionResources::network()
{
    return m_network;
}

}  // namespace srf::internal::resources

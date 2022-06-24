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

#include "internal/resources/manager.hpp"

#include "internal/system/partitions.hpp"
#include "internal/system/system.hpp"

#include "srf/internal/system/iresources.hpp"

#include <glog/logging.h>

#include <utility>

namespace srf::internal::resources {

Manager::Manager(const system::SystemProvider& system) : Manager(std::make_unique<system::Resources>(system)) {}

Manager::Manager(std::unique_ptr<system::Resources> resources) :
  SystemProvider(*resources),
  m_system(std::move(resources))
{
    // for each host partition, construct the runnable resources
    for (std::size_t i = 0; i < this->system().partitions().host_partitions().size(); ++i)
    {
        m_runnable.emplace_back(*m_system, i);
    }

    // if network, then for each flattened partition
}

runnable::Resources& Manager::runnable(std::size_t partition_id)
{
    const auto& partitions = system().partitions().flattened();
    CHECK_LT(partition_id, partitions.size());
    const auto host_id = partitions.at(partition_id).host_partition_id();
    CHECK_LT(host_id, m_runnable.size());
    return m_runnable.at(host_id);
}

std::size_t Manager::partition_count() const
{
    return system().partitions().flattened().size();
};

std::size_t Manager::device_count() const
{
    return system().partitions().device_partitions().size();
};

}  // namespace srf::internal::resources

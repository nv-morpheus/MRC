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

#include "internal/resources/system_resources.hpp"

#include "internal/system/host_partition.hpp"
#include "internal/system/partitions.hpp"
#include "internal/system/system.hpp"

#include <glog/logging.h>

namespace srf::internal::resources {

std::shared_ptr<SystemResources> SystemResources::create(std::shared_ptr<system::System> system)
{
    return std::make_shared<SystemResources>(std::move(system));
}

SystemResources::SystemResources(std::shared_ptr<System> system) : m_system(std::move(system))
{
    CHECK(m_system);

    for (const auto& partition : m_system->partitions().host_partitions())
    {
        // Main TaskQueues
        // Memory SystemResources
        // Engine Factories
        // Launch Control
        // Host Memory Resource (not yet implemented)
        // Block Memory Cache
        auto host_resources = std::make_shared<HostResources>(m_system, partition);
        m_host_resources.push_back(host_resources);

        for (const auto& device_partition_id : partition.device_partition_ids())
        {
            // Device Memory Resource
            // Block Memory Cacher
            const auto& device_partition = m_system->partitions().device_partitions().at(device_partition_id);
            auto device_resources        = std::make_shared<DeviceResources>(device_partition, host_resources);
            m_device_resources.push_back(device_resources);
        }
    }
}

const std::vector<std::shared_ptr<HostResources>>& SystemResources::host_resources() const
{
    return m_host_resources;
}

const std::vector<std::shared_ptr<DeviceResources>>& SystemResources::device_resources() const
{
    return m_device_resources;
}

system::System& SystemResources::system() const
{
    DCHECK(m_system);
    return *m_system;
}
}  // namespace srf::internal::resources

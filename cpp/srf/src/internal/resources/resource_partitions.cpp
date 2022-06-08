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

#include "internal/resources/resource_partitions.hpp"

#include "internal/resources/device_resources.hpp"
#include "internal/resources/host_resources.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/system/host_partition.hpp"

#include <glog/logging.h>

#include <cstdint>
#include <ext/alloc_traits.h>
#include <map>

namespace srf::internal::resources {

ResourcePartitions::ResourcePartitions(std::shared_ptr<system::System> system) : SystemResources(std::move(system))
{
    std::uint32_t partition_id = 0;

    // simple map to sort the device reources by cuda_device_id
    std::map<int, size_t> cuda_to_dr;
    for (std::size_t i = 0; i < device_resources().size(); i++)
    {
        cuda_to_dr[device_resources().at(i)->cuda_device_id()] = i;
    }

    for (const auto& [cuda_id, resource_id] : cuda_to_dr)
    {
        std::shared_ptr<DeviceResources> dr = device_resources().at(resource_id);
        m_resources.push_back(std::make_shared<PartitionResources>(partition_id++, dr->host_shared(), dr));
    }

    CHECK_EQ(m_resources.size(), device_resources().size());

    for (const auto& host_resources : host_resources())
    {
        if (host_resources->partition().device_partition_ids().empty())
        {
            m_resources.push_back(std::make_shared<PartitionResources>(partition_id++, host_resources, nullptr));
        }
    }
}

std::size_t ResourcePartitions::partitions() const
{
    return m_resources.size();
}
std::size_t ResourcePartitions::gpu_count() const
{
    return device_resources().size();
}
PartitionResources& ResourcePartitions::partition(std::size_t partition_id) const
{
    DCHECK_LT(partition_id, m_resources.size());
    return *m_resources.at(partition_id);
}
std::shared_ptr<PartitionResources> ResourcePartitions::partition_shared(std::size_t partition_id) const
{
    DCHECK_LT(partition_id, m_resources.size());
    return m_resources.at(partition_id);
}
}  // namespace srf::internal::resources

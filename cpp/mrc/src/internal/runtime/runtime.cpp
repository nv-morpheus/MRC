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

#include "internal/runtime/runtime.hpp"

#include "internal/resources/manager.hpp"
#include "internal/runtime/partition.hpp"

#include "mrc/types.hpp"

#include <glog/logging.h>

#include <utility>

namespace mrc::internal::runtime {

Runtime::Runtime(std::unique_ptr<resources::Manager> resources) : m_resources(std::move(resources))
{
    CHECK(m_resources);
    for (int i = 0; i < m_resources->partition_count(); i++)
    {
        m_partitions.push_back(std::make_unique<Partition>(m_resources->partition(i)));
    }
}

Runtime::~Runtime()
{
    // the problem is that m_partitions goes away, then m_resources is destroyed
    // when not all Publishers/Subscribers which were created with a ref to a Partition
    // might not yet be finished
    m_resources->shutdown().get();
}

resources::Manager& Runtime::resources() const
{
    CHECK(m_resources);
    return *m_resources;
}
std::size_t Runtime::partition_count() const
{
    return m_partitions.size();
}
std::size_t Runtime::gpu_count() const
{
    return resources().device_count();
}
Partition& Runtime::partition(std::size_t partition_id)
{
    DCHECK_LT(partition_id, m_resources->partition_count());
    DCHECK(m_partitions.at(partition_id));
    return *m_partitions.at(partition_id);
}
}  // namespace mrc::internal::runtime

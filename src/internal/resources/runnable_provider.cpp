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

#include "internal/resources/runnable_provider.hpp"

namespace srf::internal::resources {

PartitionResourceBase::PartitionResourceBase(runnable::Resources& runnable,
                                             std::size_t partition_id,
                                             std::shared_ptr<srf::memory::memory_resource> host_mr) :
  system::PartitionProvider(runnable, partition_id),
  m_runnable(runnable),
  m_raw_host_memory_resource(std::move(host_mr))
{
    CHECK_EQ(m_runnable.host_partition_id(), partition().host_partition_id());
    CHECK(m_raw_host_memory_resource);
}
runnable::Resources& PartitionResourceBase::runnable()
{
    return m_runnable;
}

std::shared_ptr<srf::memory::memory_resource> PartitionResourceBase::raw_host_mr() const
{
    return m_raw_host_memory_resource;
}

}  // namespace srf::internal::resources

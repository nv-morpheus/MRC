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

#include "internal/resources/partition_resources_base.hpp"

#include "internal/system/partition.hpp"

#include <glog/logging.h>

namespace srf::internal::resources {

PartitionResourceBase::PartitionResourceBase(runnable::Resources& runnable, std::size_t partition_id) :
  system::PartitionProvider(runnable, partition_id),
  m_runnable(runnable)
{
    CHECK_EQ(runnable.host_partition_id(), partition().host_partition_id());
}
runnable::Resources& PartitionResourceBase::runnable()
{
    return m_runnable;
}

}  // namespace srf::internal::resources

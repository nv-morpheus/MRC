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

#include "internal/pipeline/resources.hpp"

#include <srf/metrics/registry.hpp>

#include <glog/logging.h>

#include <utility>

namespace srf::internal::pipeline {

Resources::Resources(std::shared_ptr<resources::ResourcePartitions> resources) :
  m_resources(std::move(resources)),
  m_metrics_registry(std::make_unique<metrics::Registry>())
{}
resources::ResourcePartitions& Resources::resources() const
{
    DCHECK(m_resources);
    return *m_resources;
}
metrics::Registry& Resources::metrics_registry() const
{
    DCHECK(m_metrics_registry);
    return *m_metrics_registry;
}
resources::PartitionResources& Resources::partition(std::size_t partition_id) const
{
    CHECK(m_resources);
    return m_resources->partition(partition_id);
}
std::shared_ptr<resources::PartitionResources> Resources::partition_shared(std::size_t partition_id) const
{
    CHECK(m_resources);
    return m_resources->partition_shared(partition_id);
}
}  // namespace srf::internal::pipeline

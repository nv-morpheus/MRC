/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/runtime/partition_runtime.hpp"

#include "internal/async_service.hpp"
#include "internal/codable/codable_storage.hpp"
#include "internal/network/resources.hpp"
#include "internal/pubsub/publisher_round_robin.hpp"
#include "internal/pubsub/subscriber_service.hpp"
#include "internal/remote_descriptor/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/runtime/segments_manager.hpp"

#include "mrc/pubsub/api.hpp"
#include "mrc/utils/string_utils.hpp"

#include <glog/logging.h>

#include <optional>
#include <ostream>

namespace mrc::internal::runtime {

PartitionRuntime::PartitionRuntime(Runtime& system_runtime, size_t partition_id) :
  AsyncService(MRC_CONCAT_STR("PartitionRuntime[" << partition_id << "]")),
  m_system_runtime(system_runtime),
  m_partition_id(partition_id),
  m_resources(system_runtime.resources().partition(partition_id))
{
    if (resources().network())
    {
        m_remote_descriptor_manager = std::make_shared<remote_descriptor::Manager>(resources().network()->instance_id(),
                                                                                   resources());
    }
}

PartitionRuntime::~PartitionRuntime()
{
    if (m_remote_descriptor_manager)
    {
        m_remote_descriptor_manager->service_stop();
        m_remote_descriptor_manager->service_await_join();
    }
}

size_t PartitionRuntime::partition_id() const
{
    return m_partition_id;
}

resources::PartitionResources& PartitionRuntime::resources()
{
    return m_resources;
}

control_plane::Client& PartitionRuntime::control_plane() const
{
    return m_system_runtime.control_plane();
}

PipelinesManager& PartitionRuntime::pipelines_manager() const
{
    return m_system_runtime.pipelines_manager();
}

metrics::Registry& PartitionRuntime::metrics_registry() const
{
    return m_system_runtime.metrics_registry();
}

remote_descriptor::Manager& PartitionRuntime::remote_descriptor_manager()
{
    CHECK(m_remote_descriptor_manager);
    return *m_remote_descriptor_manager;
}

std::unique_ptr<mrc::codable::ICodableStorage> PartitionRuntime::make_codable_storage()
{
    return std::make_unique<codable::CodableStorage>(m_resources);
}

runnable::RunnableResources& PartitionRuntime::runnable()
{
    return m_resources.runnable();
}

void PartitionRuntime::do_service_start(std::stop_token stop_token)
{
    m_segments_manager = std::make_unique<SegmentsManager>(*this);

    // Start the child service
    this->child_service_start(*m_segments_manager);

    this->mark_started();
}

std::shared_ptr<mrc::pubsub::IPublisherService> PartitionRuntime::make_publisher_service(
    const std::string& name,
    const mrc::pubsub::PublisherPolicy& policy)
{
    if (policy == mrc::pubsub::PublisherPolicy::RoundRobin)
    {
        return std::shared_ptr<pubsub::PublisherRoundRobin>(new pubsub::PublisherRoundRobin(name, *this));
    }

    LOG(FATAL) << "PublisherPolicy not implemented";
    return nullptr;
}

std::shared_ptr<mrc::pubsub::ISubscriberService> PartitionRuntime::make_subscriber_service(const std::string& name)
{
    return std::shared_ptr<pubsub::SubscriberService>(new pubsub::SubscriberService(name, *this));
}

}  // namespace mrc::internal::runtime

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

#include "internal/runtime/partition.hpp"

#include "internal/codable/codable_storage.hpp"
#include "internal/network/resources.hpp"
#include "internal/pubsub/publisher_round_robin.hpp"
#include "internal/pubsub/subscriber_service.hpp"
#include "internal/remote_descriptor/manager.hpp"

#include <glog/logging.h>

#include <optional>
#include <ostream>

namespace mrc::internal::runtime {

Partition::Partition(resources::PartitionResources& resources) : m_resources(resources)
{
    if (resources.network())
    {
        m_remote_descriptor_manager =
            std::make_shared<remote_descriptor::Manager>(resources.network()->instance_id(), resources);
    }
}

Partition::~Partition()
{
    if (m_remote_descriptor_manager)
    {
        m_remote_descriptor_manager->service_stop();
        m_remote_descriptor_manager->service_await_join();
    }
}

resources::PartitionResources& Partition::resources()
{
    return m_resources;
}
remote_descriptor::Manager& Partition::remote_descriptor_manager()
{
    CHECK(m_remote_descriptor_manager);
    return *m_remote_descriptor_manager;
}

std::shared_ptr<mrc::pubsub::IPublisherService> Partition::make_publisher_service(
    const std::string& name, const mrc::pubsub::PublisherPolicy& policy)
{
    if (policy == mrc::pubsub::PublisherPolicy::RoundRobin)
    {
        return std::shared_ptr<pubsub::PublisherRoundRobin>(new pubsub::PublisherRoundRobin(name, *this));
    }

    LOG(FATAL) << "PublisherPolicy not implemented";
    return nullptr;
}

std::shared_ptr<mrc::pubsub::ISubscriberService> Partition::make_subscriber_service(const std::string& name)
{
    return std::shared_ptr<pubsub::SubscriberService>(new pubsub::SubscriberService(name, *this));
}

std::unique_ptr<mrc::codable::ICodableStorage> Partition::make_codable_storage()
{
    return std::make_unique<codable::CodableStorage>(m_resources);
}

}  // namespace mrc::internal::runtime

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

#pragma once

#include "internal/pubsub/forward.hpp"
#include "internal/remote_descriptor/manager.hpp"
#include "internal/remote_descriptor/remote_descriptor.hpp"
#include "internal/resources/forward.hpp"
#include "internal/resources/manager.hpp"

#include "srf/runtime/api.hpp"
#include "srf/runtime/forward.hpp"
#include "srf/utils/macros.hpp"

#include <iterator>
#include <memory>

namespace srf::internal::runtime {

class Partition final : public srf::runtime::IPartition
{
  public:
    Partition(resources::PartitionResources& resources);
    ~Partition() final;

    DELETE_COPYABILITY(Partition);
    DELETE_MOVEABILITY(Partition);

    resources::PartitionResources& resources();

    // IPartition -> IRemoteDescriptorManager& is covariant
    remote_descriptor::Manager& remote_descriptor_manager() final;

    std::shared_ptr<pubsub::Publisher> make_publisher(const std::string& name,
                                                      const srf::pubsub::PublisherPolicy& policy);

    std::shared_ptr<pubsub::Subscriber> make_subscriber(const std::string& name);

  private:
    // IPartiton -> shared_ptr<IPublisher> is not covariant with shared_ptr<Publisher>
    // however the two are convertable, so we do this in two stages rather than directly
    std::shared_ptr<srf::pubsub::IPublisher> create_publisher_service(const std::string& name,
                                                                      const srf::pubsub::PublisherPolicy& policy) final;

    // IPartiton -> shared_ptr<ISubscriber> is not covariant with shared_ptr<Subscriber>
    // however the two are convertable, so we do this in two stages rather than directly
    std::shared_ptr<srf::pubsub::ISubscriber> create_subscriber_service(const std::string& name) final;

    resources::PartitionResources& m_resources;
    std::shared_ptr<remote_descriptor::Manager> m_remote_descriptor_manager;
};

}  // namespace srf::internal::runtime

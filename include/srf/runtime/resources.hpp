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

#include "srf/codable/api.hpp"
#include "srf/pubsub/api.hpp"
#include "srf/pubsub/publisher.hpp"
#include "srf/pubsub/publisher_policy.hpp"
#include "srf/pubsub/subscriber.hpp"

#include <string>

namespace srf::runtime {

class IResources
{
  public:
    ~IResources() = default;

    template <typename T>
    std::shared_ptr<pubsub::Publisher<T>> make_publisher(std::string name, pubsub::PublisherPolicy policy)
    {
        return {new pubsub::Publisher<T>(await_create_publisher_service(name, policy))};
    }

    template <typename T>
    std::shared_ptr<pubsub::Subscriber<T>> make_subscriber(std::string name)
    {
        auto subscriber = std::shared_ptr<pubsub::Subscriber<T>>(new pubsub::Subscriber<T>());
        auto service    = await_create_subscriber_service(subscriber.m_decoder);
        subscriber->attach_service(std::move(service));
        return subscriber;
    }

  private:
    virtual std::unique_ptr<pubsub::IPublisher> await_create_publisher_service(
        const std::string& name, const pubsub::PublisherPolicy& policy) = 0;

    virtual std::unique_ptr<pubsub::IPublisher> await_create_subscriber_service(
        std::function<void(std::unique_ptr<codable::IDecodableStorage>)> decoder) = 0;
};

}  // namespace srf::runtime

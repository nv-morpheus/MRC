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

#pragma once

#include "internal/control_plane/client.hpp"
#include "internal/control_plane/client/instance.hpp"
#include "internal/control_plane/client/subscription_service.hpp"
#include "internal/data_plane/request.hpp"
#include "internal/data_plane/server.hpp"
#include "internal/expected.hpp"
#include "internal/network/resources.hpp"
#include "internal/pubsub/pub_sub_base.hpp"
#include "internal/pubsub/subscriber.hpp"
#include "internal/resources/forward.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/service.hpp"

#include "srf/channel/channel.hpp"
#include "srf/channel/ingress.hpp"
#include "srf/channel/status.hpp"
#include "srf/codable/encode.hpp"
#include "srf/codable/encoded_object.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/operators/router.hpp"
#include "srf/node/queue.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/node/source_properties.hpp"
#include "srf/protos/architect.pb.h"
#include "srf/utils/macros.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

namespace srf::internal::pubsub {

class SubscriberManagerBase : public PubSubBase
{
  public:
    SubscriberManagerBase(std::string name, resources::PartitionResources& resources) :
      PubSubBase(std::move(name), resources.network()->control_plane()),
      m_resources(resources)
    {}

    ~SubscriberManagerBase() override = default;

    const std::string& role() const final
    {
        return role_subscriber();
    }

    const std::set<std::string>& subscribe_to_roles() const final
    {
        static std::set<std::string> r = {};
        return r;
    }

  protected:
    inline resources::PartitionResources& resources()
    {
        return m_resources;
    }

  private:
    resources::PartitionResources& m_resources;
};

template <typename T>
class Subscriber;

template <typename T>
class SubscriberManager : public SubscriberManagerBase
{
  public:
    SubscriberManager(std::string name, resources::PartitionResources& resources) :
      SubscriberManagerBase(std::move(name), resources)
    {}

    ~SubscriberManager() override
    {
        service_stop();
        service_await_join();
    }

    Future<std::shared_ptr<Subscriber<T>>> make_subscriber()
    {
        return m_subscriber_promise.get_future();
    }

  private:
    void update_tagged_instances(const std::string& role,
                                 const std::unordered_map<std::uint64_t, InstanceID>& tagged_instances) final
    {
        LOG(FATAL) << "pubsub::Subscriber should never get TaggedInstance updates";
    }

    void do_service_start() override
    {
        SubscriptionService::do_service_start();

        CHECK(this->tag() != 0);

        auto drop_subscription_service_lambda = drop_subscription_service();

        auto subscriber = std::shared_ptr<Subscriber<T>>(new Subscriber<T>(service_name(), this->tag()),
                                                         [drop_subscription_service_lambda](Subscriber<T>* ptr) {
                                                             drop_subscription_service_lambda();
                                                             delete ptr;
                                                         });

        auto network_reader = std::make_unique<node::RxSink<memory::TransientBuffer>>(
            [](memory::TransientBuffer buffer) { LOG(FATAL) << "SubscriberManager network_reader - implement me"; });
        node::make_edge(resources().network()->data_plane().server().deserialize_source().source(this->tag()),
                        *network_reader);

        auto launch_options = resources().network()->control_plane().client().launch_options();

        m_reader = resources()
                       .runnable()
                       .launch_control()
                       .prepare_launcher(launch_options, std::move(network_reader))
                       ->ignition();

        m_subscriber = subscriber;
        m_subscriber_promise.set_value(std::move(subscriber));

        SRF_THROW_ON_ERROR(activate_subscription_service());
    }

    void do_service_await_live() override
    {
        m_reader->await_live();
    }

    void do_service_stop() override
    {
        resources().network()->data_plane().server().deserialize_source().drop_edge(this->tag());
        m_reader->stop();
    }

    void do_service_kill() override
    {
        resources().network()->data_plane().server().deserialize_source().drop_edge(this->tag());
        m_reader->kill();
    }

    void do_service_await_join() override
    {
        m_reader->await_join();
    }

    std::weak_ptr<Subscriber<T>> m_subscriber;
    std::unique_ptr<srf::runnable::Runner> m_reader;
    Promise<std::shared_ptr<Subscriber<T>>> m_subscriber_promise;
};

template <typename T>
std::shared_ptr<Subscriber<T>> make_subscriber(const std::string& name, resources::PartitionResources& resources)
{
    auto manager = std::make_unique<SubscriberManager<T>>(name, resources);
    auto future  = manager->make_subscriber();
    resources.network()->control_plane().register_subscription_service(std::move(manager));
    return future.get();
}

}  // namespace srf::internal::pubsub

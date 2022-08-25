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
#include "internal/data_plane/client.hpp"
#include "internal/data_plane/request.hpp"
#include "internal/expected.hpp"
#include "internal/network/resources.hpp"
#include "internal/pubsub/pub_sub_base.hpp"
#include "internal/pubsub/publisher.hpp"
#include "internal/resources/forward.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/service.hpp"
#include "internal/ucx/common.hpp"

#include "srf/channel/channel.hpp"
#include "srf/channel/ingress.hpp"
#include "srf/channel/status.hpp"
#include "srf/codable/encode.hpp"
#include "srf/codable/encoded_object.hpp"
#include "srf/node/edge_builder.hpp"
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

class PublisherManagerBase : public PubSubBase
{
  public:
    PublisherManagerBase(std::string name, resources::PartitionResources& resources) :
      PubSubBase(std::move(name), resources.network()->control_plane()),
      m_resources(resources)
    {}

    ~PublisherManagerBase() override = default;

    const std::string& role() const final
    {
        return role_publisher();
    }

    const std::set<std::string>& subscribe_to_roles() const final
    {
        static std::set<std::string> r = {role_subscriber()};
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
class PublisherManager : public PublisherManagerBase
{
  public:
    PublisherManager(std::string name, resources::PartitionResources& resources) :
      PublisherManagerBase(std::move(name), resources)
    {}

    ~PublisherManager() override
    {
        service_await_join();
    }

    Future<std::shared_ptr<Publisher<T>>> make_publisher()
    {
        return m_publisher_promise.get_future();
    }

  protected:
    const std::unordered_map<std::uint64_t, InstanceID>& tagged_instances() const
    {
        return m_tagged_instances;
    }

    const std::unordered_map<std::uint64_t, std::shared_ptr<ucx::Endpoint>>& tagged_endpoints() const
    {
        return m_tagged_endpoints;
    }

  private:
    virtual void write(T&& object) = 0;
    virtual void on_update()       = 0;

    void update_tagged_instances(const std::string& role,
                                 const std::unordered_map<std::uint64_t, InstanceID>& tagged_instances) final
    {
        DCHECK_EQ(role, role_subscriber());

        // todo - convert tagged instances -> tagged endpoints
        m_tagged_instances = tagged_instances;
        for (const auto& [tag, instance_id] : m_tagged_instances)
        {
            m_tagged_endpoints[tag] = resources().network()->data_plane().client().endpoint_shared(instance_id);
        }
        on_update();
    }

    void do_service_start() override
    {
        SubscriptionService::do_service_start();

        CHECK(this->tag() != 0);

        auto drop_subscription_service_lambda = drop_subscription_service();

        auto publisher = std::shared_ptr<Publisher<T>>(new Publisher<T>(service_name(), this->tag()),
                                                       [drop_subscription_service_lambda](Publisher<T>* ptr) {
                                                           drop_subscription_service_lambda();
                                                           delete ptr;
                                                       });
        auto sink      = std::make_unique<srf::node::RxSink<T>>([this](T data) { write(std::move(data)); });
        srf::node::make_edge(*publisher, *sink);

        auto launch_options = resources().network()->control_plane().client().launch_options();
        m_writer =
            resources().runnable().launch_control().prepare_launcher(launch_options, std::move(sink))->ignition();

        m_publisher = publisher;
        m_publisher_promise.set_value(std::move(publisher));

        SRF_THROW_ON_ERROR(activate_subscription_service());
    }

    void do_service_await_live() override
    {
        m_writer->await_live();
    }

    void do_service_stop() override
    {
        m_writer->stop();
    }

    void do_service_kill() override
    {
        m_writer->kill();
    }

    void do_service_await_join() override
    {
        m_writer->await_join();
    }

    std::weak_ptr<Publisher<T>> m_publisher;
    std::unique_ptr<srf::runnable::Runner> m_writer;
    std::unordered_map<std::uint64_t, InstanceID> m_tagged_instances;
    std::unordered_map<std::uint64_t, std::shared_ptr<ucx::Endpoint>> m_tagged_endpoints;
    std::unordered_map<std::uint64_t, InstanceID>::const_iterator m_next;
    Promise<std::shared_ptr<Publisher<T>>> m_publisher_promise;
};

template <typename T>
class PublisherRoundRobin : public PublisherManager<T>
{
  public:
    using PublisherManager<T>::PublisherManager;

  private:
    void on_update() final
    {
        m_next = this->tagged_endpoints().cbegin();
    }

    void write(T&& object) final
    {
        DCHECK(this->resources().runnable().main().caller_on_same_thread());

        while (this->tagged_instances().empty())
        {
            // await subscribers
            // for now just return and drop the object
            return;
        }

        data_plane::RemoteDescriptorMessage msg;

        msg.tag      = m_next->first;
        msg.endpoint = m_next->second;

        if (++m_next == this->tagged_endpoints().cend())
        {
            m_next = this->tagged_endpoints().cbegin();
        }

        std::size_t val = 42;

        msg.rd = this->resources().network()->remote_descriptor_manager().register_object(std::move(object));
        CHECK(this->resources().network()->data_plane().client().remote_descriptor_channel().await_write(
                  std::move(msg)) == channel::Status::success);
    }

    std::unordered_map<std::uint64_t, std::shared_ptr<ucx::Endpoint>>::const_iterator m_next;
};

enum class PublisherType
{
    RoundRobin,
};

template <typename T>
std::shared_ptr<Publisher<T>> make_publisher(const std::string& name,
                                             PublisherType type,
                                             resources::PartitionResources& resources)
{
    std::unique_ptr<PublisherManager<T>> manager;
    switch (type)
    {
    case PublisherType::RoundRobin:
        manager = std::make_unique<PublisherRoundRobin<T>>(name, resources);
        break;
    default:
        LOG(FATAL) << "unknown publisher type";
    }
    CHECK(manager);

    auto future = manager->make_publisher();
    resources.network()->control_plane().register_subscription_service(std::move(manager));
    return future.get();
}

}  // namespace srf::internal::pubsub

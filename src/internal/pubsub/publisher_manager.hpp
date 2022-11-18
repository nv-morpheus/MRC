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
#include "internal/remote_descriptor/manager.hpp"
#include "internal/resources/forward.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/runtime/runtime.hpp"
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

class PublisherBackend : public PubSubBase
{
  public:
    using element_type = std::unique_ptr<srf::codable::EncodedStorage>;

    PublisherBackend(std::string name, runtime::Runtime& runtime);

    ~PublisherBackend() override;

    Future<std::unique_ptr<Publisher>> make_publisher()
    {
        return m_publisher_promise.get_future();
    }

    const std::string& role() const final;

    const std::set<std::string>& subscribe_to_roles() const final;

  protected:
    const std::unordered_map<std::uint64_t, InstanceID>& tagged_instances() const;

    const std::unordered_map<std::uint64_t, std::shared_ptr<ucx::Endpoint>>& tagged_endpoints() const;

  private:
    virtual void write(element_type&& data) = 0;
    virtual void on_update()                = 0;

    void update_tagged_instances(const std::string& role,
                                 const std::unordered_map<std::uint64_t, InstanceID>& tagged_instances) final
    {
        DCHECK_EQ(role, role_subscriber());

        // todo - convert tagged instances -> tagged endpoints
        m_tagged_instances = tagged_instances;
        for (const auto& [tag, instance_id] : m_tagged_instances)
        {
            // m_tagged_endpoints[tag] = resources().network()->data_plane().client().endpoint_shared(instance_id);
        }
        on_update();
    }

    void do_service_start() override;
    void do_service_await_live() override;
    void do_service_stop() override;
    void do_service_kill() override;
    void do_service_await_join() override;

    std::unique_ptr<srf::runnable::Runner> m_writer;
    std::unordered_map<std::uint64_t, InstanceID> m_tagged_instances;
    std::unordered_map<std::uint64_t, std::shared_ptr<ucx::Endpoint>> m_tagged_endpoints;
    Promise<std::unique_ptr<Publisher>> m_publisher_promise;
};

template <typename T>
class PublisherRoundRobin : public PublisherBackend
{
  public:
    using PublisherBackend::PublisherBackend;

  private:
    void on_update() final
    {
        m_next = this->tagged_endpoints().cbegin();
    }

    void write(T&& object) final
    {
        LOG(INFO) << "publisher writing object";

        DCHECK(this->resources().runnable().main().caller_on_same_thread());

        while (this->tagged_instances().empty())
        {
            // await subscribers
            // for now just return and drop the object
            boost::this_fiber::yield();
        }

        data_plane::RemoteDescriptorMessage msg;

        msg.tag      = m_next->first;
        msg.endpoint = m_next->second;

        if (++m_next == this->tagged_endpoints().cend())
        {
            m_next = this->tagged_endpoints().cbegin();
        }

        msg.rd = this->runtime().remote_descriptor_manager().register_encoded_object(std::move(object));
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
std::unique_ptr<Publisher> make_publisher(const std::string& name, PublisherType type, runtime::Runtime& runtime)
{
    std::unique_ptr<PublisherBackend> backend;

    switch (type)
    {
    case PublisherType::RoundRobin:
        backend = std::make_unique<PublisherRoundRobin<T>>(name, runtime);
        break;
    default:
        LOG(FATAL) << "unknown publisher type";
    }
    CHECK(backend);

    auto future = backend->make_publisher();
    runtime.resources().network()->control_plane().register_subscription_service(std::move(backend));
    return future.get();
}

}  // namespace srf::internal::pubsub

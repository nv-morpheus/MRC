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

#include "srf/channel/ingress.hpp"
#include "srf/codable/encoded_object.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/operators/operator.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/pubsub/api.hpp"
#include "srf/pubsub/subscription_service.hpp"
#include "srf/runtime/forward.hpp"

namespace srf::pubsub {

template <typename T>
class Publisher final : public SubscriptionService, public node::Operator<T>, public channel::Ingress<T>
{
  public:
    ~Publisher() final
    {
        stop();
        await_join();
    }

    // Ingress<T> overrides

    // publish T by capturing it as an encoded object, then pushing that encoded object to the internal publisher
    inline channel::Status await_write(T&& data) final
    {
        auto encoded_object = codable::EncodedObject<T>::create(std::move(data), m_service->create_storage());
        return m_service->await_write(std::move(encoded_object));
    }

  private:
    Publisher(std::unique_ptr<IPublisher> publisher) : m_service(std::move(publisher))
    {
        CHECK(m_service);

        // form a persistent connection to the operator
        m_persistent_channel = std::make_unique<srf::node::SourceChannelWriteable<T>>();
        srf::node::make_edge(*m_persistent_channel, *this);
    }

    // SubscriptionService overrides

    ISubscriptionService& service() const final
    {
        return *m_service;
    }

    // issuing a stop should not stop the service immediately if there are upstream connections; since we hold an
    // upstream connection as the m_persistent_channel, we simply release the persistent channel on stop, then the issue
    // a SuscriptionService::stop() final upstream disconnect.
    void stop() final
    {
        m_persistent_channel.reset();
    }

    // Operator<T> overrides

    // forward the operator pass thru write to the publicly exposed await_write method
    channel::Status on_next(T&& data) final
    {
        return await_write(std::move(data));
    }

    // the trigger of this method signifies that all upstream connections, including the locally held persistent
    // connection, have been released. this should be the signal to initiate a stop on the service, as a stop on the
    // Publisher<T> has already been initiated.
    void on_complete() final
    {
        SubscriptionService::stop();
    }

    // Member Variables

    // internal implementation of publisher
    const std::unique_ptr<IPublisher> m_service;

    // this holds the operator open;
    std::unique_ptr<srf::node::SourceChannelWriteable<T>> m_persistent_channel;

    friend runtime::IResources;
};

}  // namespace srf::pubsub

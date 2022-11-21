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
#include "srf/codable/decode.hpp"
#include "srf/codable/encoded_object.hpp"
#include "srf/control_plane/subscription_service_forwarder.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/operators/operator_component.hpp"
#include "srf/node/queue.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/pubsub/api.hpp"
#include "srf/runtime/forward.hpp"
#include "srf/runtime/remote_descriptor.hpp"

namespace srf::pubsub {

/**
 * @brief Receiving end of the Publisher->Subscriber chain
 *
 * Subscriber<T> is a source of T where T is received from the data plane as described in the data path below.
 * Subscriber has a template specialization, i.e. Subscriber<RemoteDescriptor> which bypasses the decoding of the object
 * and provides direct acess to the remote descriptor. This is useful for propagating the RemoteDescriptor without
 * having to pull required bulk data held on the remote instance.
 *
 * This object is always created as as shared_ptr with a copy held by the instance of the control plane on a specific
 * partition whose resources are using for encoding, decoding and data transport. After edges are formed, this object
 * can be destroyed and its lifecycle will be properly managed by the runtime.
 *
 * Publisher<T> Data Path:
 * [T] -> EncodedObject<T> -> EncodedStorage -> RemoteDescriptor -> Transient Buffer -> Data Plane Tagged Send
 *
 * Subscriber<T> Data Path:
 * Data Plane Tagged Received -> Transient Buffer -> RemoteDescriptor -> Subscriber/Source<T> ->
 *
 * Subscriber<RemoteDescriptor> Data Path:
 * Data Plane Tagged Received -> Transient Buffer -> RemoteDescriptor -> Subscriber/Source<RemoteDescriptor> ->
 */
template <typename T>
class Subscriber final : public node::Queue<T>,
                         public control_plane::SubscriptionServiceForwarder,
                         private srf::node::SourceChannelWriteable<T>
{
  public:
    ~Subscriber() final
    {
        request_stop();
        await_join();
    }

    /**
     * @brief Start the Subscriber
     *
     * Unlike the Publisher which is at the start of the PubSub chain, the Subscriber is at the end of the chain; this
     * means it must be fully connected before it is started so that the data path is complete.
     *
     * We form edges are part of the await_start override to allow customization of the Queue prior to start.
     */
    void await_start() final
    {
        if (!is_startable())
        {
            LOG(FATAL) << "attempting to start a service which not startable";
            return;
        }

        // Edge - SourceChannelWritable<T> -> Queue<T>
        srf::node::SourceChannelWriteable<T>& typed_source = *this;
        srf::node::make_edge(typed_source, *this);

        // Edge - IPublisher -> OperatorComponent
        m_rd_sink = std::make_shared<node::OperatorComponent<srf::runtime::RemoteDescriptor>>(
            // on_next
            [this](srf::runtime::RemoteDescriptor&& rd) {
                auto obj = rd.decode<T>();
                return srf::node::SourceChannelWriteable<T>::await_write(std::move(obj));
            },
            // on_complete
            [this] { srf::node::SourceChannelWriteable<T>::release_channel(); });
        srf::node::make_edge(*m_service, *m_rd_sink);

        // After the edges have been formed, we have a complete pipeline from the data plane to a channel. If we started
        // the service prior to the edge construction, we might get data flowing through an incomplete operator chain
        // and result in an inablility to handle backpressure.
        m_service->await_start();
    }

  private:
    Subscriber(std::shared_ptr<ISubscriber> service) : m_service(std::move(service))
    {
        CHECK(m_service);
    }

    ISubscriptionService& service() const final
    {
        CHECK(m_service);
        return *m_service;
    }

    std::shared_ptr<ISubscriber> m_service;
    std::shared_ptr<node::OperatorComponent<srf::runtime::RemoteDescriptor>> m_rd_sink;

    friend runtime::IPartition;
};

/**
 * @brief Template specialization of Subscriber for RemoteDescriptor
 *
 * @tparam T
 */
template <>
class Subscriber<srf::runtime::RemoteDescriptor> final : public node::Queue<runtime::RemoteDescriptor>,
                                                         public control_plane::SubscriptionServiceForwarder
{
  public:
    ~Subscriber() final
    {
        request_stop();
        await_join();
    }

    /**
     * @brief Start the Subscriber
     *
     * Unlike the Publisher which is at the start of the PubSub chain, the Subscriber is at the end of the chain; this
     * means it must be fully connected before it is started so that the data path is complete.
     *
     * We form edges are part of the await_start override to allow customization of the Queue prior to start.
     */
    void await_start() final
    {
        if (!is_startable())
        {
            LOG(FATAL) << "attempting to start a service which not startable";
            return;
        }

        // Edge - IPublisher -> Queue
        srf::node::make_edge(*m_service, *this);

        // After the edges have been formed, we have a complete pipeline from the data plane to a channel. If we started
        // the service prior to the edge construction, we might get data flowing through an incomplete operator chain
        // and result in an inablility to handle backpressure.
        m_service->await_start();
    }

  private:
    Subscriber(std::shared_ptr<ISubscriber> service) : m_service(std::move(service)) {}

    ISubscriptionService& service() const final
    {
        CHECK(m_service);
        return *m_service;
    }

    std::shared_ptr<ISubscriber> m_service;

    friend runtime::IPartition;
};

}  // namespace srf::pubsub

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

#include "mrc/codable/api.hpp"
#include "mrc/codable/decode.hpp"
#include "mrc/codable/encoded_object.hpp"
#include "mrc/control_plane/subscription_service_forwarder.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/operators/operator_component.hpp"
#include "mrc/node/queue.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/pubsub/api.hpp"
#include "mrc/runtime/api.hpp"
#include "mrc/runtime/remote_descriptor.hpp"
#include "mrc/utils/macros.hpp"

namespace mrc::pubsub {

/**
 * @brief Receiving end of the Publisher->Subscriber chain
 *
 * Subscriber<T> is a source of T where T is received from the data plane as described in the data path below.
 * Subscriber has a template specialization, i.e. Subscriber<RemoteDescriptor> which bypasses the decoding of the object
 * and provides direct acess to the remote descriptor. This is useful for propagating the RemoteDescriptor without
 * having to pull required bulk data held on the remote instance.
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
                         private mrc::node::SourceChannelWriteable<T>
{
  public:
    static std::unique_ptr<Subscriber> create(std::string name, runtime::IPartition& partition)
    {
        return std::unique_ptr<Subscriber>{new Subscriber(partition.make_subscriber_service(name))};
    }

    ~Subscriber() final
    {
        request_stop();
        await_join();
    }

    DELETE_COPYABILITY(Subscriber);
    DELETE_MOVEABILITY(Subscriber);

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

        LOG(INFO) << "forming first edge: typed_source -> self [Node<T>]";

        // Edge - SourceChannelWritable<T> -> Queue<T>
        mrc::node::SourceChannelWriteable<T>& typed_source = *this;
        mrc::node::make_edge(typed_source, *this);

        LOG(INFO) << "forming second edge: IPublisherService -> operator_component";

        // Edge - IPublisherService -> OperatorComponent
        m_rd_sink = std::make_shared<node::OperatorComponent<mrc::runtime::RemoteDescriptor>>(
            // on_next
            [this](mrc::runtime::RemoteDescriptor&& rd) {
                auto obj = rd.decode<T>();
                return mrc::node::SourceChannelWriteable<T>::await_write(std::move(obj));
            },
            // on_complete
            [this] { mrc::node::SourceChannelWriteable<T>::release_channel(); });
        mrc::node::make_edge(*m_service, *m_rd_sink);

        // After the edges have been formed, we have a complete pipeline from the data plane to a channel. If we started
        // the service prior to the edge construction, we might get data flowing through an incomplete operator chain
        // and result in an inablility to handle backpressure.
        LOG(INFO) << "issuing isubscriber::await_start";
        m_service->await_start();
        LOG(INFO) << "subscriber::await_start finished";
    }

  private:
    Subscriber(std::shared_ptr<ISubscriberService> service) : m_service(std::move(service))
    {
        CHECK(m_service);
    }

    ISubscriptionService& service() const final
    {
        CHECK(m_service);
        return *m_service;
    }

    std::shared_ptr<ISubscriberService> m_service;
    std::shared_ptr<node::OperatorComponent<mrc::runtime::RemoteDescriptor>> m_rd_sink;

    friend runtime::IPartition;
};

/**
 * @brief Template specialization of Subscriber for RemoteDescriptor
 *
 * @tparam T
 */
template <>
class Subscriber<mrc::runtime::RemoteDescriptor> final : public node::Queue<runtime::RemoteDescriptor>,
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

        // Edge - IPublisherService -> Queue
        mrc::node::make_edge(*m_service, *this);

        // After the edges have been formed, we have a complete pipeline from the data plane to a channel. If we started
        // the service prior to the edge construction, we might get data flowing through an incomplete operator chain
        // and result in an inablility to handle backpressure.
        m_service->await_start();
    }

  private:
    Subscriber(std::shared_ptr<ISubscriberService> service) : m_service(std::move(service)) {}

    ISubscriptionService& service() const final
    {
        CHECK(m_service);
        return *m_service;
    }

    std::shared_ptr<ISubscriberService> m_service;

    friend runtime::IPartition;
};

}  // namespace mrc::pubsub

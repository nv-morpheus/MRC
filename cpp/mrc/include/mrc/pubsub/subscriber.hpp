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

#pragma once

#include "mrc/codable/api.hpp"
#include "mrc/codable/decode.hpp"
#include "mrc/codable/encoded_object.hpp"
#include "mrc/control_plane/subscription_service_forwarder.hpp"
#include "mrc/edge/edge_builder.hpp"
#include "mrc/node/generic_sink.hpp"
#include "mrc/node/operators/node_component.hpp"
#include "mrc/node/queue.hpp"
#include "mrc/node/readable_endpoint.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_channel_owner.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/node/writable_entrypoint.hpp"
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
class Subscriber final : public control_plane::SubscriptionServiceForwarder,
                         public node::ReadableProvider<T>,
                         private node::WritableProvider<mrc::runtime::RemoteDescriptor>
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

        m_persistent_channel = std::make_shared<mrc::node::ReadableEndpoint<T>>();
        mrc::make_edge(*this, *m_persistent_channel);

        // Make a connection from this to the service
        mrc::make_edge<ISubscriberService, node::WritableProvider<mrc::runtime::RemoteDescriptor>>(*m_service, *this);

        // LOG(INFO) << "forming first edge: typed_source -> self [Node<T>]";

        // // Edge - SourceChannelWritable<T> -> Queue<T>
        // mrc::node::WritableEntrypoint<T>& typed_source = *this;
        // mrc::make_edge(typed_source, *this);

        // LOG(INFO) << "forming second edge: IPublisherService -> operator_component";

        // // Edge - IPublisherService -> OperatorComponent
        // m_rd_sink = std::make_shared<node::LambdaSinkComponent<mrc::runtime::RemoteDescriptor>>(
        //     // on_next
        //     [this](mrc::runtime::RemoteDescriptor&& rd) {
        //         auto obj = rd.decode<T>();
        //         return mrc::node::WritableEntrypoint<T>::await_write(std::move(obj));
        //     },
        //     // on_complete
        //     [this] { mrc::node::WritableEntrypoint<T>::release_edge_connection(); });
        // mrc::make_edge(*m_service, *m_rd_sink);

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

        // Create the internal channel
        edge::EdgeChannel<mrc::runtime::RemoteDescriptor> edge_channel(
            std::make_unique<mrc::channel::BufferedChannel<mrc::runtime::RemoteDescriptor>>());

        // Wrap the upstream with a converting edge to an encoded object
        auto downstream_edge = std::make_shared<edge::LambdaConvertingEdgeReadable<mrc::runtime::RemoteDescriptor, T>>(
            [this](mrc::runtime::RemoteDescriptor&& rd) {
                // Perform the decode
                return rd.decode<T>();
            },
            edge_channel.get_reader());

        node::SinkProperties<mrc::runtime::RemoteDescriptor>::init_owned_edge(edge_channel.get_writer());
        node::SourceProperties<T>::init_owned_edge(downstream_edge);
    }

    ISubscriptionService& service() const final
    {
        CHECK(m_service);
        return *m_service;
    }

    std::shared_ptr<ISubscriberService> m_service;
    // std::shared_ptr<node::LambdaSinkComponent<mrc::runtime::RemoteDescriptor>> m_rd_sink;

    // this holds the operator open;
    std::shared_ptr<mrc::node::ReadableEndpoint<T>> m_persistent_channel;

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
        mrc::make_edge(*m_service, *this);

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

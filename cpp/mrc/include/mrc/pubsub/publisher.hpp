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

#include "mrc/channel/ingress.hpp"
#include "mrc/codable/encoded_object.hpp"
#include "mrc/control_plane/subscription_service_forwarder.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/operators/unique_operator.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/pubsub/api.hpp"
#include "mrc/runtime/api.hpp"
#include "mrc/runtime/remote_descriptor.hpp"
#include "mrc/utils/macros.hpp"

namespace mrc::pubsub {

/**
 * @brief Publishes an object T which will be received by one more more Subscribers.
 *
 * This object is both directly writeable, but also connectable to multiple upstream sources of T. By its nature as an
 * operator, forward progress is performed not by a progress engine, but rather by the callers of await_write or by the
 * execution context driving forward progress along the edge.
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
class Publisher final : public control_plane::SubscriptionServiceForwarder,
                        public node::UniqueOperator<T>,
                        public channel::Ingress<T>
{
  public:
    static std::unique_ptr<Publisher> create(std::string name,
                                             const PublisherPolicy& policy,
                                             runtime::IPartition& partition)
    {
        return std::unique_ptr<Publisher>{new Publisher(partition.make_publisher_service(name, policy))};
    }

    ~Publisher() final
    {
        request_stop();
        await_join();
    }

    DELETE_COPYABILITY(Publisher);
    DELETE_MOVEABILITY(Publisher);

    // [Ingress<T>] publish T by capturing it as an encoded object, then pushing that encoded object to the internal
    // publisher
    channel::Status await_write(T&& data) final;

    void await_start() final
    {
        // form a persistent connection to the operator
        // data flowing in from operator edges are forwarded to the public await_write
        m_persistent_channel = std::make_unique<mrc::node::SourceChannelWriteable<T>>();
        mrc::node::make_edge(*m_persistent_channel, *this);

        CHECK(m_service);
        m_service->await_start();
    }

    void request_stop() final
    {
        // drop the persistent channel holding keeping the operator live
        // when the last connection is dropped, the Operator<T>::on_complete override will be triggered
        m_persistent_channel.reset();
    }

  private:
    Publisher(std::shared_ptr<IPublisherService> publisher) : m_service(std::move(publisher))
    {
        CHECK(m_service);
    }

    // [ISubscriptionServiceControl] - this overrides the SubscriptionServiceForwarder forwarding method
    // issuing a request to stop should only happen after all edges have been dropped from this operator
    // we simply release the persistent channel on stop, then the issue
    // a SuscriptionService::stop() final upstream disconnect.

    // [Operator<T>] the trigger of this method signifies that all upstream connections, including the locally held
    // persistent connection, have been released. this should be the signal to initiate a stop on the service, as a stop
    // on the Publisher<T> has already been initiated.
    void on_complete() final
    {
        request_stop();
    }

    // [Operator<T>] forward the operator pass thru write to the publicly exposed await_write method
    channel::Status on_next(T&& data) final
    {
        return await_write(std::move(data));
    }

    // [SubscriptionServiceForwarder] access storage
    ISubscriptionService& service() const final
    {
        return *m_service;
    }

    // internal type-erased implementation of publisher
    const std::shared_ptr<IPublisherService> m_service;

    // this holds the operator open;
    std::unique_ptr<mrc::node::SourceChannelWriteable<T>> m_persistent_channel;

    friend runtime::IPartition;
};

template <typename T>
channel::Status Publisher<T>::await_write(T&& data)
{
    auto encoded_object = codable::EncodedObject<T>::create(std::move(data), m_service->create_storage());
    return m_service->publish(std::move(encoded_object));
}

// template specialization for remote descriptors

// template <>
// channel::Status Publisher<runtime::RemoteDescriptor>::await_write(runtime::RemoteDescriptor&& rd)
// {
//     return m_service->publish(std::move(rd));
// }

}  // namespace mrc::pubsub

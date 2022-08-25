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

#include "internal/data_plane/client.hpp"
#include "internal/remote_descriptor/messages.hpp"
#include "internal/remote_descriptor/remote_descriptor.hpp"
#include "internal/remote_descriptor/storage.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/runnable/engines.hpp"
#include "internal/service.hpp"
#include "internal/ucx/resources.hpp"

#include "srf/channel/buffered_channel.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/types.hpp"

#include <map>
#include <memory>

namespace srf::internal::remote_descriptor {

/**
 * @brief Creates and Manages RemoteDescriptors
 *
 * The remote descriptor Manager is responsible for transforming an object to a remote descriptor. The Manager will take
 * ownership of the object and hold it until the all RemoteDescriptor (RD) reference count tokens are released.
 *
 * The manager is also responsible for decrement the global reference count when a remote descriptor is released. This
 * is done via a ucx active message.
 *
 * This object will register an active message handler with the data plane's ucx worker. The registered callback will be
 * triggered and executed by the thread running the ucx worker progress engine, i.e. the data plane's io thread. To
 * avoid any potentially latency heavy operations occurring on the data plane io thread will push a message over a
 * channel back to a handler running on the main task queue to perform the decrement and any potential release of the
 * storaged object.
 *
 * The shutdown sequence should be:
 *  1. detatch the active message handler function from the ucx runtime
 *  2. close the decrement channel
 *  3. await on the decrement handler executing on main
 */
class Manager final : private resources::PartitionResourceBase,
                      private Service,
                      public std::enable_shared_from_this<Manager>
{
  public:
    Manager(const InstanceID& instance_id, ucx::Resources& ucx, data_plane::Client& client) :
      resources::PartitionResourceBase(ucx),
      m_instance_id(instance_id),
      m_ucx(ucx),
      m_client(client)
    {
        service_start();
        service_await_live();
    }

    ~Manager()
    {
        Service::call_in_destructor();
    }

    template <typename T>
    RemoteDescriptor register_object(T&& object)
    {
        return store_object(TypedStorage<T>::create(std::move(object)));
    }

    RemoteDescriptor take_ownership(std::unique_ptr<const srf::codable::protos::RemoteDescriptor> rd)
    {
        auto non_const_rd = std::unique_ptr<srf::codable::protos::RemoteDescriptor>(
            const_cast<srf::codable::protos::RemoteDescriptor*>(rd.release()));
        return RemoteDescriptor(shared_from_this(), std::move(non_const_rd));
    }

    std::size_t size() const;

  private:
    static std::uint32_t active_message_id();

    RemoteDescriptor store_object(std::unique_ptr<Storage> object);

    void decrement_tokens(std::unique_ptr<const srf::codable::protos::RemoteDescriptor> rd);
    void decrement_tokens(std::size_t object_id, std::size_t token_count);

    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;

    // <object_id, storage>
    std::map<std::size_t, std::unique_ptr<Storage>> m_stored_objects;
    const InstanceID m_instance_id;

    ucx::Resources& m_ucx;
    data_plane::Client& m_client;
    std::unique_ptr<srf::runnable::Runner> m_decrement_handler;
    std::unique_ptr<srf::node::SourceChannelWriteable<RemoteDescriptorDecrementMessage>> m_decrement_channel;

    std::mutex m_mutex;

    friend RemoteDescriptor;
    friend network::Resources;
};

}  // namespace srf::internal::remote_descriptor

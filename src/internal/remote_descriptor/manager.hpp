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

#include "internal/remote_descriptor/messages.hpp"
#include "internal/remote_descriptor/storage.hpp"
#include "internal/resources/forward.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/service.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/codable/api.hpp"
#include "mrc/codable/encoded_object.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/protos/codable.pb.h"
#include "mrc/runnable/runner.hpp"
#include "mrc/runtime/remote_descriptor.hpp"
#include "mrc/runtime/remote_descriptor_handle.hpp"
#include "mrc/runtime/remote_descriptor_manager.hpp"
#include "mrc/types.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

namespace mrc::internal::remote_descriptor {

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
class Manager final : private Service,
                      public std::enable_shared_from_this<Manager>,
                      public mrc::runtime::IRemoteDescriptorManager
{
  public:
    Manager(const InstanceID& instance_id, resources::PartitionResources& resources);

    ~Manager() override;

    mrc::runtime::RemoteDescriptor make_remote_descriptor(mrc::codable::protos::RemoteDescriptor&& proto);
    mrc::runtime::RemoteDescriptor make_remote_descriptor(
        std::unique_ptr<mrc::runtime::IRemoteDescriptorHandle> handle);

    std::size_t size() const;

    InstanceID instance_id() const;

    const mrc::codable::IDecodableStorage& encoding(const std::size_t& object_id) const;

    void release_handle(std::unique_ptr<mrc::runtime::IRemoteDescriptorHandle> handle) final;

    static std::unique_ptr<mrc::runtime::IRemoteDescriptorHandle> unwrap_handle(mrc::runtime::RemoteDescriptor&& rd)
    {
        return rd.release_handle();
    }

    mrc::runtime::RemoteDescriptor register_encoded_object(std::unique_ptr<mrc::codable::EncodedStorage> object) final;

  private:
    static std::uint32_t active_message_id();

    std::unique_ptr<mrc::codable::ICodableStorage> create_storage() final;

    void decrement_tokens(std::size_t object_id, std::size_t token_count);

    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;

    // <object_id, storage>
    std::map<std::size_t, Storage> m_stored_objects;
    const InstanceID m_instance_id;

    resources::PartitionResources& m_resources;
    std::unique_ptr<mrc::runnable::Runner> m_decrement_handler;
    std::unique_ptr<mrc::node::SourceChannelWriteable<RemoteDescriptorDecrementMessage>> m_decrement_channel;

    mutable std::mutex m_mutex;

    friend internal::runtime::Partition;
};

}  // namespace mrc::internal::remote_descriptor

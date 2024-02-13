/**
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

#include "internal/control_plane/state/root_state.hpp"
#include "internal/runtime/runtime_provider.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/core/async_service.hpp"
#include "mrc/edge/edge_readable.hpp"
#include "mrc/node/generic_sink.hpp"
#include "mrc/runtime/remote_descriptor.hpp"
#include "mrc/types.hpp"

#include <cstddef>
#include <map>
#include <memory>
#include <stop_token>

namespace ucxx {
class Endpoint;
}

namespace mrc::codable {
class EncodedStorage;
}  // namespace mrc::codable
namespace mrc::edge {
template <typename T>
class IWritableAcceptor;
template <typename T>
class IWritableProvider;
}  // namespace mrc::edge
namespace mrc::node {
template <typename T>
class Queue;

template <typename KeyT, typename ValueT>
class TaggedRouter;
}  // namespace mrc::node

namespace mrc::runtime {
class Descriptor;

class DataPlaneSystemManager : public AsyncService, public InternalRuntimeProvider
{
  public:
    DataPlaneSystemManager(IInternalRuntimeProvider& runtime);
    ~DataPlaneSystemManager() override;

    std::string get_ucx_address() const;

    // This is what each ingress object will connect to in order to pull the next message
    std::shared_ptr<edge::IReadableProvider<std::unique_ptr<ValueDescriptor>>> get_readable_ingress_channel(
        PortAddress2 port_address) const;

    // This is what local and remote egress objects will connect to in order to push the next message to a local ingress
    std::shared_ptr<edge::IWritableProvider<std::unique_ptr<ValueDescriptor>>> get_writable_ingress_channel(
        PortAddress2 port_address) const;

    // This is what each egress object will connect to in order to push messages to remote ingress objects
    std::shared_ptr<edge::IWritableProvider<std::unique_ptr<ValueDescriptor>>> get_writable_egress_channel(
        PortAddress2 port_address) const;

    std::shared_ptr<node::Queue<std::unique_ptr<Descriptor>>> get_incoming_port_channel(InstanceID port_address) const;
    std::shared_ptr<edge::IWritableProvider<std::unique_ptr<Descriptor>>> get_outgoing_port_channel(
        InstanceID port_address) const;

  private:
    void do_service_start(std::stop_token stop_token) override;

    void process_state_update(const control_plane::state::ControlPlaneState& state);

    channel::Status send_descriptor(PortAddress2 port_destination, std::unique_ptr<ValueDescriptor>&& descriptor);

    // control_plane::state::ControlPlaneState m_previous_state;

    mutable Mutex m_port_mutex;

    std::unique_ptr<data_plane::DataPlaneResources2> m_resources;

    std::unique_ptr<node::TaggedRouter<PortAddress2, std::unique_ptr<ValueDescriptor>>> m_inbound_dispatcher;

    std::map<PortAddress2, std::weak_ptr<node::Queue<std::unique_ptr<ValueDescriptor>>>> m_ingress_port_channels;
    std::map<PortAddress2, std::weak_ptr<node::LambdaSinkComponent<std::unique_ptr<ValueDescriptor>>>>
        m_egress_port_channels;

    std::map<InstanceID, std::weak_ptr<node::Queue<std::unique_ptr<Descriptor>>>> m_incoming_port_channels;
    std::map<InstanceID, std::weak_ptr<node::Queue<std::unique_ptr<Descriptor>>>> m_outgoing_port_channels;
};

class DataPlaneManager : public AsyncService, public InternalRuntimeProvider
{
  public:
    DataPlaneManager(IInternalRuntimeProvider& runtime, size_t partition_id);
    ~DataPlaneManager() override;

    std::shared_ptr<edge::IWritableProvider<codable::EncodedStorage>> get_output_channel(SegmentAddress address);
    std::shared_ptr<edge::IWritableAcceptor<codable::EncodedStorage>> get_input_channel(SegmentAddress address);

    void sync_state(const control_plane::state::Worker& worker);

  private:
    void do_service_start(std::stop_token stop_token) override;
};

}  // namespace mrc::runtime

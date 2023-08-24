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

#include "mrc/core/async_service.hpp"
#include "mrc/types.hpp"

#include <cstddef>
#include <map>
#include <memory>
#include <stop_token>

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
}  // namespace mrc::node

namespace mrc::runtime {
class Descriptor;

class DataPlaneSystemManager : public AsyncService, public InternalRuntimeProvider
{
  public:
    DataPlaneSystemManager(IInternalRuntimeProvider& runtime);
    ~DataPlaneSystemManager() override;

    std::shared_ptr<node::Queue<std::unique_ptr<Descriptor>>> get_incoming_port_channel(InstanceID port_address) const;
    std::shared_ptr<edge::IWritableProvider<std::unique_ptr<Descriptor>>> get_outgoing_port_channel(
        InstanceID port_address) const;

  private:
    void do_service_start(std::stop_token stop_token) override;

    void process_state_update(const control_plane::state::ControlPlaneState& state);

    // control_plane::state::ControlPlaneState m_previous_state;

    mutable Mutex m_port_mutex;
    std::map<InstanceID, std::weak_ptr<node::Queue<std::unique_ptr<Descriptor>>>> m_incoming_port_channels;
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

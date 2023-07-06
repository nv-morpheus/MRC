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

#include "internal/control_plane/client.hpp"
#include "internal/control_plane/state/root_state.hpp"
#include "internal/remote_descriptor/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/runnable/runnable_resources.hpp"
#include "internal/runtime/resource_manager_base.hpp"
#include "internal/runtime/runtime_provider.hpp"
#include "internal/segment/segment_instance.hpp"
#include "internal/ucx/ucx_resources.hpp"

#include "mrc/codable/encoded_object.hpp"
#include "mrc/core/async_service.hpp"
#include "mrc/edge/forward.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/queue.hpp"
#include "mrc/runtime/remote_descriptor.hpp"
#include "mrc/types.hpp"

#include <cstddef>
#include <memory>
#include <optional>

namespace mrc::memory {
class DeviceResources;
class HostResources;
}  // namespace mrc::memory
namespace mrc::network {
class NetworkResources;
}  // namespace mrc::network
namespace mrc::runnable {
class RunnableResources;
}  // namespace mrc::runnable

namespace mrc::runtime {

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

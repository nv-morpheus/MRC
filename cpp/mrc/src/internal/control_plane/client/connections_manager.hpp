/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/control_plane/client/state_manager.hpp"
#include "internal/ucx/common.hpp"

#include "mrc/node/writable_entrypoint.hpp"
#include "mrc/protos/architect.pb.h"
#include "mrc/types.hpp"

#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace mrc::internal::control_plane {
class Client;
}  // namespace mrc::internal::control_plane
namespace mrc::internal::ucx {
class UcxResources;
}  // namespace mrc::internal::ucx

namespace mrc::internal::control_plane::client {
class Instance;

class ConnectionsManager : public StateManager
{
  public:
    using update_channel_t = mrc::node::WritableEntrypoint<const protos::StateUpdate>;

    ConnectionsManager(Client& client, update_channel_t& update_channel);
    ~ConnectionsManager() override;

    std::map<InstanceID, std::unique_ptr<client::Instance>> register_ucx_addresses(
        std::vector<std::optional<ucx::UcxResources>>& ucx_resources);

    const MachineID& machine_id() const;
    const std::vector<InstanceID>& instance_ids() const;

    const std::map<InstanceID, MachineID>& locality_map() const;
    const std::map<InstanceID, ucx::WorkerAddress>& worker_addresses() const;
    const std::map<InstanceID, std::unique_ptr<update_channel_t>>& instance_channels() const;

  private:
    void do_update(const protos::StateUpdate&& update_msg) final;
    void do_connections_update(const protos::UpdateConnectionsState& connections);
    void do_route_state_update(const protos::StateUpdate&& update_msg);

    MachineID m_machine_id;
    std::vector<InstanceID> m_instance_ids;
    std::map<InstanceID, MachineID> m_locality_map;
    std::map<InstanceID, ucx::WorkerAddress> m_worker_addresses;
    std::map<InstanceID, std::unique_ptr<update_channel_t>> m_update_channels;
};

}  // namespace mrc::internal::control_plane::client

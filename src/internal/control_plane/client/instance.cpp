/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/control_plane/client/instance.hpp"

#include "internal/control_plane/client.hpp"
#include "internal/data_plane/client.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/utils/contains.hpp"

#include "srf/node/edge_builder.hpp"
#include "srf/protos/architect.pb.h"

namespace srf::internal::control_plane::client {

Instance::Instance(Client& client,
                   InstanceID instance_id,
                   resources::PartitionResourceBase& base,
                   srf::node::SourceChannel<const protos::StateUpdate>& update_channel) :
  resources::PartitionResourceBase(base),
  m_client(client),
  m_instance_id(instance_id)
{
    auto update_handler = std::make_unique<srf::node::RxSink<protos::StateUpdate>>(
        [this](protos::StateUpdate update) { do_handle_state_update(update); });

    srf::node::make_edge(update_channel, *update_handler);

    m_update_handler =
        runnable().launch_control().prepare_launcher(client.launch_options(), std::move(update_handler))->ignition();
}

Instance::~Instance()
{
    DVLOG(10) << "client instance: " << m_instance_id << " issuing drop request";
    protos::TaggedInstance msg;
    msg.set_instance_id(m_instance_id);
    CHECK(client().await_unary<protos::Ack>(protos::ClientUnaryDropWorker, std::move(msg)));

    // requesting an update to avoid the timeout
    client().request_update();

    // this should block until an update is issued by the server for the client to finalize the instance drop
    m_update_handler->await_join();
    DVLOG(10) << "client instance: " << m_instance_id << " dropped by server - shutting down client-side";
}

void Instance::do_handle_state_update(const protos::StateUpdate& update)
{
    DVLOG(10) << "control plane instance on partition " << partition_id() << " got an update for "
              << update.service_name();

    // std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    if (update.has_subscription_service())
    {
        // do subscription update
    }
}

const InstanceID& Instance::instance_id() const
{
    return m_instance_id;
}

Client& Instance::client()
{
    return m_client;
}

// void Instance::attach_data_plane_client(data_plane::Client* data_plane)
// {
//     CHECK(data_plane);
//     std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//     CHECK(m_data_plane == nullptr);
//     m_data_plane = data_plane;
//     for (const auto& [id, address] : m_worker_addresses)
//     {
//         m_data_plane->register_instance(id, address);
//     }
// }

}  // namespace srf::internal::control_plane::client

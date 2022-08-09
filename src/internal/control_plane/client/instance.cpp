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
#include "internal/resources/partition_resources_base.hpp"

#include "srf/node/edge_builder.hpp"
#include "srf/protos/architect.pb.h"

namespace srf::internal::control_plane::client {

Instance::Instance(Client& client,
                   InstanceID instance_id,
                   resources::PartitionResourceBase& base,
                   srf::node::SourceChannel<protos::StateUpdate>& update_channel) :
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
    m_client.drop_instance(m_instance_id);
    m_update_handler->await_join();
    DVLOG(10) << "client instance " << m_instance_id << " completed";
}

void Instance::do_handle_state_update(const protos::StateUpdate& update) {}

const InstanceID& Instance::instance_id() const
{
    return m_instance_id;
}
Client& Instance::client()
{
    return m_client;
}
}  // namespace srf::internal::control_plane::client

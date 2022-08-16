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

#include "internal/control_plane/client/state_manager.hpp"

#include "internal/control_plane/client.hpp"
#include "internal/resources/partition_resources_base.hpp"

#include "srf/channel/forward.hpp"
#include "srf/channel/recent_channel.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/protos/architect.pb.h"

#include <glog/logging.h>

namespace srf::internal::control_plane::client {

StateManager::StateManager(Client& client, node::SourceChannel<const protos::StateUpdate>& update_channel) :
  m_client(client)
{
    auto sink = std::make_unique<node::RxSink<protos::StateUpdate>>(
        [this](protos::StateUpdate update_msg) { update(std::move(update_msg)); });
    // sink->update_channel(std::make_unique<channel::RecentChannel<protos::StateUpdate>>(1));
    node::make_edge(update_channel, *sink);
    m_runner =
        client.runnable().launch_control().prepare_launcher(client.launch_options(), std::move(sink))->ignition();
}

StateManager::~StateManager()
{
    m_runner->await_join();
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    if (!m_update_promises.empty())
    {
        auto eptr = std::make_exception_ptr(Error::create("StateManager being destroyed with outstanding promises"));
        for (auto& p : m_update_promises)
        {
            p.set_exception(eptr);
        }
    }
}

Future<void> StateManager::update_future()
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return m_update_promises.emplace_back().get_future();
}

void StateManager::update(const protos::StateUpdate&& update_msg)
{
    if (m_nonce < update_msg.nonce())
    {
        m_nonce = update_msg.nonce();
        do_update(std::move(update_msg));
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        for (auto& p : m_update_promises)
        {
            p.set_value();
        }
        m_update_promises.clear();
    }
}

const Client& StateManager::client() const
{
    return m_client;
}

Client& StateManager::client()
{
    return m_client;
}

}  // namespace srf::internal::control_plane::client

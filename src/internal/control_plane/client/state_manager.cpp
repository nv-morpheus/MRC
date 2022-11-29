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
#include "internal/expected.hpp"
#include "internal/runnable/resources.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/protos/architect.pb.h"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launcher.hpp"

#include <boost/fiber/future/promise.hpp>
#include <ext/alloc_traits.h>
#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <algorithm>
#include <exception>
#include <utility>

namespace mrc::internal::control_plane::client {

StateManager::StateManager(Client& client) : m_client(client) {}

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

void StateManager::start_with_channel(node::SourceChannel<const protos::StateUpdate>& update_channel)
{
    auto sink = std::make_unique<node::RxSink<protos::StateUpdate>>(
        [this](protos::StateUpdate update_msg) { update(std::move(update_msg)); });
    // sink->update_channel(std::make_unique<channel::RecentChannel<protos::StateUpdate>>(1));
    node::make_edge(update_channel, *sink);
    m_runner =
        client().runnable().launch_control().prepare_launcher(client().launch_options(), std::move(sink))->ignition();
}

void StateManager::await_join()
{
    CHECK(m_runner);
    m_runner->await_join();
}
}  // namespace mrc::internal::control_plane::client

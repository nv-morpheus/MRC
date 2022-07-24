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

#pragma once

#include "srf/node/edge_builder.hpp"
#include "srf/node/operators/broadcast.hpp"
#include "srf/node/sink_channel.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/protos/architect.pb.h"

#include <set>
#include <string>

namespace srf::internal::control_plane::client {

class SubscriptionService final
{
  public:
    SubscriptionService(std::string name, std::set<std::string> roles) :
      m_name(std::move(name)),
      m_roles(std::move(roles))
    {
        m_bcast = std::make_shared<srf::node::Broadcast<protos::SubscriptionServiceUpdate>>();
        srf::node::make_edge(m_channel, *m_bcast);
    }

    const std::string& name() const
    {
        return m_name;
    }

    const std::set<std::string>& roles() const
    {
        return m_roles;
    }

    srf::channel::Status await_write(protos::SubscriptionServiceUpdate&& message)
    {
        return m_channel.await_write(std::move(message));
    }

    void add_subscriber(srf::node::SinkChannel<protos::SubscriptionServiceUpdate>& subscriber)
    {
        srf::node::make_edge(*m_bcast, subscriber);
    }

  private:
    std::string m_name;
    std::set<std::string> m_roles;
    srf::node::SourceChannelWriteable<protos::SubscriptionServiceUpdate> m_channel;
    std::shared_ptr<srf::node::Broadcast<protos::SubscriptionServiceUpdate>> m_bcast;
};

}  // namespace srf::internal::control_plane::client

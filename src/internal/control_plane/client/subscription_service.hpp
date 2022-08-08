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

#include "srf/channel/status.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/operators/router.hpp"
#include "srf/node/sink_channel.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/protos/architect.pb.h"
#include "srf/types.hpp"

#include <set>
#include <string>

namespace srf::internal::control_plane::client {

class SubscriptionService final
{
    using router_t = srf::node::Router<InstanceID, protos::ServiceUpdate>;

  public:
    SubscriptionService(std::string name, std::set<std::string> roles) :
      m_name(std::move(name)),
      m_roles(std::move(roles))
    {
        m_router = std::make_shared<router_t>();
        srf::node::make_edge(m_channel, *m_router);
    }

    const std::string& name() const
    {
        return m_name;
    }

    const std::set<std::string>& roles() const
    {
        return m_roles;
    }

    srf::channel::Status await_write(const InstanceID& instance_id, protos::ServiceUpdate&& message)
    {
        if (m_router->has_edge(instance_id))
        {
            return m_channel.await_write(std::make_pair(instance_id, std::move(message)));
        }
        return channel::Status::error;
    }

    void add_instance(const InstanceID& instance_id,
                      srf::node::SinkChannel<protos::SubscriptionServiceUpdate>& subscriber)
    {
        CHECK(!m_router->has_edge(instance_id));
        srf::node::make_edge(m_router->source(instance_id), subscriber);
    }

    void drop_instance(const InstanceID& instance_id)
    {
        m_router->drop_edge(instance_id);
    }

  private:
    std::string m_name;
    std::set<std::string> m_roles;
    srf::node::SourceChannelWriteable<router_t::source_data_t> m_channel;
    std::shared_ptr<router_t> m_router;
};

}  // namespace srf::internal::control_plane::client

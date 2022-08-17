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

#include "internal/control_plane/client/state_manager.hpp"
#include "internal/service.hpp"

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

class Instance;

class SubscriptionService
{
  public:
    SubscriptionService(std::string service_name, std::set<std::string> roles) :
      m_service_name(std::move(service_name)),
      m_roles(std::move(roles))
    {}
    virtual ~SubscriptionService() = default;

    const std::string& service_name() const
    {
        return m_service_name;
    }

    const std::set<std::string>& roles() const
    {
        return m_roles;
    }

  protected:
    enum class State
    {
        Initialzed,
        Created,
        Registered,
        Activated,
        Operational,
        Completed
    };

    void forward_state(const State& new_state)
    {
        CHECK(m_state < new_state);
        m_state = new_state;
    }

    const State& state() const
    {
        return m_state;
    }

  private:
    std::string m_service_name;
    std::set<std::string> m_roles;
    State m_state{State::Initialzed};
};

class SubscriptionServiceUpdater : public SubscriptionService, public StateManager
{
  public:
    SubscriptionServiceUpdater(std::string name,
                               std::set<std::string> roles,
                               Client& client,
                               node::SourceChannel<const protos::StateUpdate>& update_channel) :
      SubscriptionService(std::move(name), std::move(roles)),
      StateManager(client, update_channel)
    {}

  private:
    void do_update(const protos::StateUpdate&& update_msg) final;
    virtual void do_subscription_service_update(const protos::SubscriptionServiceState& update_msg) = 0;
};

}  // namespace srf::internal::control_plane::client

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
    using router_t = srf::node::Router<InstanceID, protos::StateUpdate>;

  public:
    SubscriptionService(std::string name, std::set<std::string> roles) :
      m_name(std::move(name)),
      m_roles(std::move(roles))
    {}
    virtual ~SubscriptionService() = default;

    const std::string& name() const
    {
        return m_name;
    }

    const std::set<std::string>& roles() const
    {
        return m_roles;
    }

  private:
    std::string m_name;
    std::set<std::string> m_roles;
};

class SubscriptionServiceUpdater : public SubscriptionService, private Service
{
  public:
    using SubscriptionService::SubscriptionService;

    Future<void> fence_update()
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        return m_update_promises.emplace_back().get_future();
    }

  private:
    void update(const protos::StateUpdate& update_msg)
    {
        CHECK(update_msg.has_subscription_service());
        if (m_nonce < update_msg.nonce())
        {
            m_nonce = update_msg.nonce();
            do_update(update_msg.subscription_service());
            std::lock_guard<decltype(m_mutex)> lock(m_mutex);
            for (auto& p : m_update_promises)
            {
                p.set_value();
            }
            m_update_promises.clear();
        }
    }

    virtual void do_update(const protos::SubscriptionServiceState& update_msg) = 0;

    std::size_t m_nonce{1};
    std::vector<Promise<void>> m_update_promises;
    std::mutex m_mutex;

    friend Instance;
};

}  // namespace srf::internal::control_plane::client

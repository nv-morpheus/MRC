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

#include "internal/control_plane/server/client_instance.hpp"

#include "srf/protos/architect.pb.h"
#include "srf/types.hpp"

#include <string>

namespace srf::internal::control_plane {

class Role
{
  public:
    Role(std::string service_name, std::string role_name) :
      m_service_name(std::move(service_name)),
      m_role_name(std::move(role_name))
    {}

    // subscribers are notified when new members are added
    void add_member(std::shared_ptr<server::ClientInstance> instance);
    void add_subscriber(std::shared_ptr<server::ClientInstance> instance);

  private:
    // protos::SubscriptionServiceUpdate make_update();
    protos::SubscriptionServiceUpdate make_update() const;

    static void await_update(const std::shared_ptr<server::ClientInstance>& instance,
                             const protos::SubscriptionServiceUpdate& update);

    std::string m_service_name;
    std::string m_role_name;
    std::set<std::shared_ptr<server::ClientInstance>> m_members;
    std::set<std::shared_ptr<server::ClientInstance>> m_subscribers;
    std::size_t m_nonce{0};
    Mutex m_mutex;
};

class SubscriptionService final
{
  public:
    SubscriptionService(std::string name, std::set<std::string> roles);
    ~SubscriptionService() = default;

    void register_instance(const std::string& role,
                           const std::set<std::string>& subscribe_to_roles,
                           std::shared_ptr<server::ClientInstance> instance);

    bool has_role(const std::string& role) const;
    bool compare_roles(const std::set<std::string>& roles) const;

  private:
    void add_role(const std::string& name);
    Role& get_role(const std::string& name);

    std::string m_name;

    // roles are defines at construction time in the body of the constructor
    // no new keys should be added
    std::map<std::string, std::unique_ptr<Role>> m_roles;
};

}  // namespace srf::internal::control_plane

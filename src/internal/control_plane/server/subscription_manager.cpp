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

#include "internal/control_plane/server/subscription_manager.hpp"

#include "internal/utils/contains.hpp"

#include "srf/protos/architect.pb.h"

#include <glog/logging.h>

namespace srf::internal::control_plane::server {

void Role::add_member(std::uint64_t tag, std::shared_ptr<server::ClientInstance> instance)
{
    DCHECK(!contains(m_members, tag));
    m_members[tag] = instance;
    mark_as_modified();
}

void Role::add_subscriber(std::uint64_t tag, std::shared_ptr<server::ClientInstance> instance)
{
    DCHECK(!contains(m_subscribers, tag));
    m_subscribers[tag] = instance;

    // issue one off update to new subscriber; no need to mark_as_modified since the list of entries
    // watched by the subscribers has not changed
}

void Role::drop_tag(std::uint64_t tag)
{
    if (m_members.erase(tag) != 0)
    {
        mark_as_modified();
    }
    m_subscribers.erase(tag);
}

void Role::do_issue_update(const protos::StateUpdate& update)
{
    DVLOG(10) << "issue_update for " << m_service_name << "/" << m_role_name;
    for (const auto& [tag, instance] : m_subscribers)
    {
        await_update(instance, update);
    }
}

void Role::do_make_update(protos::StateUpdate& update) const
{
    auto* service = update.mutable_subscription_service();
    service->set_role(m_role_name);
    for (const auto& [tag, instance] : m_members)
    {
        auto* tagged_instance = service->add_tagged_instances();
        tagged_instance->set_tag(tag);
        tagged_instance->set_instance_id(instance->get_id());
    }
}

void Role::await_update(const std::shared_ptr<server::ClientInstance>& instance, const protos::StateUpdate& update)
{
    protos::Event event;
    event.set_event(protos::EventType::ServerStateUpdate);
    event.set_tag(instance->get_id());
    event.mutable_message()->PackFrom(update);
    instance->stream_writer().await_write(std::move(event));
}

const std::string& Role::service_name() const
{
    return m_service_name;
}
const std::string& Role::role_name() const
{
    return m_role_name;
}

// SubscriptionService

SubscriptionService::SubscriptionService(std::string name, std::set<std::string> roles) : m_name(std::move(name))
{
    for (const auto& name : roles)
    {
        m_roles[name] = std::make_unique<Role>(m_name, name);
    }
    DCHECK_EQ(roles.size(), m_roles.size());
}

SubscriptionService::tag_t SubscriptionService::register_instance(std::shared_ptr<server::ClientInstance> instance,
                                                                  const std::string& role,
                                                                  const std::set<std::string>& subscribe_to_roles)
{
    // ensure that role is not subscribing to itself for multi-role
    if (m_roles.size() > 1)
    {
        CHECK(!contains(subscribe_to_roles, role));  // todo(cpp20) use std::set::contains
    }

    // ensure all roles and subscribe_to_roles are valid
    CHECK(contains(m_roles, role));
    for (const auto& s2r : subscribe_to_roles)
    {
        CHECK(contains(m_roles, s2r));
    }

    auto tag = register_instance_id(instance->get_id());

    get_role(role).add_member(tag, instance);
    for (const auto& s2r : subscribe_to_roles)
    {
        get_role(s2r).add_subscriber(tag, instance);
    }

    return tag;
}

bool SubscriptionService::compare_roles(const std::set<std::string>& roles) const
{
    if (m_roles.size() != roles.size())
    {
        return false;
    }
    return std::all_of(roles.begin(), roles.end(), [this](auto& role) { return contains(m_roles, role); });
}

bool SubscriptionService::has_role(const std::string& role) const
{
    auto search = m_roles.find(role);
    return search != m_roles.end();
}
Role& SubscriptionService::get_role(const std::string& name)
{
    auto search = m_roles.find(name);
    CHECK(search != m_roles.end());
    CHECK(search->second);
    return *(search->second);
}

// void SubscriptionService::drop_instance(std::shared_ptr<server::ClientInstance> instance)
// {
//     for (auto& [name, role] : m_roles)
//     {
//         role->drop_instance(instance);
//     }
// }

void SubscriptionService::do_drop_tag(const tag_t& tag)
{
    for (auto& [name, role] : m_roles)
    {
        role->drop_tag(tag);
    }
}

void SubscriptionService::do_issue_update()
{
    for (auto& [name, role] : m_roles)
    {
        role->issue_update();
    }
}

const std::string& SubscriptionService::service_name() const
{
    return m_name;
}
}  // namespace srf::internal::control_plane::server

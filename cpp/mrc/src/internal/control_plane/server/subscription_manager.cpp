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

#include "internal/grpc/stream_writer.hpp"
#include "internal/utils/contains.hpp"

#include "mrc/protos/architect.pb.h"

#include <glog/logging.h>
#include <google/protobuf/any.pb.h>

#include <algorithm>
#include <cstdint>
#include <ostream>

namespace mrc::internal::control_plane::server {

void Role::add_member(std::uint64_t tag, std::shared_ptr<server::ClientInstance> instance)
{
    DCHECK(!contains(m_members, tag));
    DVLOG(10) << "service: " << service_name() << "; role: " << role_name() << "; adding member with tag: " << tag;
    m_members[tag] = instance;
    mark_as_modified();
}

void Role::add_subscriber(std::uint64_t tag, std::shared_ptr<server::ClientInstance> instance)
{
    DCHECK(!contains(m_subscribers, tag));
    DVLOG(10) << "service: " << service_name() << "; role: " << role_name() << "; adding subscriber with tag: " << tag;
    m_subscribers[tag]       = instance;
    m_subscriber_nonces[tag] = 0;

    // optional: issue one off update to new subscriber
    // note: the global state will eventual become consistent when the server periodically evaluates and issues updates
    // auto update = make_update();
    // await_update(instance, update);
}

void Role::drop_tag(std::uint64_t tag)
{
    if (m_subscribers.erase(tag) != 0)
    {
        DVLOG(10) << "service: " << service_name() << "; role: " << role_name()
                  << "; dropping subscriber with tag: " << tag;
    }
    m_subscriber_nonces.erase(tag);

    // if a member is dropped, we mark the state as dirty and issue an updated membership list to all subscribers
    if (contains(m_members, tag))
    {
        mark_as_modified();
        m_latched_members[tag] = std::make_pair(current_nonce(), m_members.at(tag));
        m_members.erase(tag);
        // note: the dropped tag instance is still "latched" to the service, i.e. no drop request from the server will
        // be issued until all subscribers have synchronized on the membership update
        DVLOG(10) << "service: " << service_name() << "; role: " << role_name()
                  << "; latching member with tag: " << tag;
    }
    evaluate_latches();
}

void Role::update_subscriber_nonce(const std::uint64_t& tag, const std::uint64_t& nonce)
{
    auto search = m_subscriber_nonces.find(tag);
    if (search != m_subscriber_nonces.end())
    {
        DVLOG(10) << "updating subscriber with tag: " << tag << " with nonce: " << nonce;
        search->second = nonce;
    }
    evaluate_latches();
}

void Role::evaluate_latches()
{
    std::set<std::uint64_t> tags_to_remove;
    for (const auto& t_ni : m_latched_members)
    {
        const auto nonce = t_ni.second.first;
        // t_ni => <tag, <nonce, instance>>
        // evalute the nonce of latched tagged instances against the current set of subscriber nonces, i.e. the
        // state of the subscribers
        if (std::all_of(m_subscriber_nonces.begin(), m_subscriber_nonces.end(), [&nonce](const auto& tn) {
                // tn => <tag, nonce>
                return nonce <= tn.second;
            }))
        {
            const auto tag      = t_ni.first;
            const auto instance = t_ni.second.second;

            // if the nonce of the latched tag is less than or equal to the nonces of all current subscribers,
            // then we can safely drop the latched instance
            DVLOG(10) << "issuing drop request for former member with tag: " << tag << "; nonce: " << nonce;

            // issue drop request to client
            protos::StateUpdate update;
            update.set_service_name(service_name());
            update.set_instance_id(instance->get_id());
            auto* dropped = update.mutable_drop_subscription_service();
            dropped->set_role(role_name());
            dropped->set_tag(tag);
            await_update(instance, update);

            // remove tag after issue drop requests
            tags_to_remove.insert(tag);
        }
    }

    DVLOG(10) << "server service: " << service_name() << "; role: " << role_name() << "; dropping "
              << tags_to_remove.size() << " latched members";
    for (const auto& tag : tags_to_remove)
    {
        m_latched_members.erase(tag);
    }
}

bool Role::has_update() const
{
    return true;
}

void Role::do_make_update(protos::StateUpdate& update) const
{
    auto* service = update.mutable_update_subscription_service();
    service->set_role(m_role_name);
    for (const auto& [tag, instance] : m_members)
    {
        auto* tagged_instance = service->add_tagged_instances();
        tagged_instance->set_tag(tag);
        tagged_instance->set_instance_id(instance->get_id());
    }
}

void Role::do_issue_update(const protos::StateUpdate& update)
{
    DVLOG(10) << "issue_update for " << m_service_name << "/" << m_role_name;
    std::set<std::uint64_t> unique_instances;
    for (const auto& [tag, instance] : m_subscribers)
    {
        if (!contains(unique_instances, instance->get_id()))
        {
            await_update(instance, update);
            unique_instances.insert(instance->get_id());
        }
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

Expected<TagID> SubscriptionService::register_instance(std::shared_ptr<server::ClientInstance> instance,
                                                       const std::string& role,
                                                       const std::set<std::string>& subscribe_to_roles)
{
    // ensure all roles and subscribe_to_roles are valid
    MRC_CHECK(contains(m_roles, role));
    for (const auto& s2r : subscribe_to_roles)
    {
        MRC_CHECK(contains(m_roles, s2r));
    }

    auto tag = register_instance_id(instance->get_id());

    return tag;
}

Expected<> SubscriptionService::activate_instance(std::shared_ptr<server::ClientInstance> instance,
                                                  const std::string& role,
                                                  const std::set<std::string>& subscribe_to_roles,
                                                  TagID tag)
{
    // ensure all roles and subscribe_to_roles are valid
    MRC_CHECK(contains(m_roles, role));
    for (const auto& s2r : subscribe_to_roles)
    {
        MRC_CHECK(contains(m_roles, s2r));
    }

    MRC_CHECK(is_issued_tag(tag));

    get_role(role).add_member(tag, instance);
    for (const auto& s2r : subscribe_to_roles)
    {
        get_role(s2r).add_subscriber(tag, instance);
    }
    return {};
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

void SubscriptionService::do_drop_tag(const TagID& tag)
{
    for (auto& [name, role] : m_roles)
    {
        DVLOG(10) << "do_drop_tag: " << tag << "; for " << name;
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

Expected<> SubscriptionService::update_role(const protos::UpdateSubscriptionServiceRequest& update_req)
{
    MRC_CHECK(update_req.service_name() == service_name());
    auto search = m_roles.find(update_req.role());
    MRC_CHECK(search != m_roles.end());
    for (const auto& tag : update_req.tags())
    {
        search->second->update_subscriber_nonce(tag, update_req.nonce());
    }
    return {};
}

}  // namespace mrc::internal::control_plane::server

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
#include "internal/control_plane/server/tagged_issuer.hpp"
#include "internal/control_plane/server/versioned_issuer.hpp"
#include "internal/expected.hpp"

#include "mrc/protos/architect.pb.h"
#include "mrc/types.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

namespace mrc::internal::control_plane::server {

class Role;

/**
 * @brief A specialize TaggedManager to synchronize tag and instance_id information across between a collection of
 * client-side objects with common linkages, e.g. the Publisher/Subscriber services which form the building blocks for
 * Ingress/EgressPorts use instances of SubscriptionService for Publishers to get control plane updates to the list of
 * Subscribers.
 *
 * The PubSub example is specialzied example of the more generic SubscriptionService. This example has two roles:
 * {"publisher", "subscriber"}, where the publisher gets updates on the subscriber role, but the subscribers only
 * register as members and do not receive publisher updates.
 *
 * The client must create a subscription service with the globally unique service_name and the set of roles. Once
 * created, instances of Role (e.g. Publisher/Subscriber) can be registered. Registration provides a globally unique
 * 64-bit tag to uniquely identify the instance of the particular Role.
 *
 * Registered instances are not live until they are activated. This allows the client-side to attach any listeners on
 * the provided registration tag. Once activated, updates will be issued periodically or on a client request.
 *
 * To provide consensus, issued updates must report back after the update is applied client-side. This enables the
 * server to track the state of the clients. Using the PubSub as an example, this client-side state tracking is
 * necessary to ensure that a Subscriber is not deactivated until all Publishers have updated their internal states to
 * reflect the removal of the Subscriber. When a Subscriber requests to be dropped, it enters a latched state.
 */
class SubscriptionService final : public TaggedIssuer
{
  public:
    SubscriptionService(std::string name, std::set<std::string> roles);
    ~SubscriptionService() final = default;

    const std::string& service_name() const final;

    bool has_role(const std::string& role) const;
    bool compare_roles(const std::set<std::string>& roles) const;

    Expected<TagID> register_instance(std::shared_ptr<server::ClientInstance> instance,
                                      const std::string& role,
                                      const std::set<std::string>& subscribe_to_roles);

    Expected<> activate_instance(std::shared_ptr<server::ClientInstance> instance,
                                 const std::string& role,
                                 const std::set<std::string>& subscribe_to_roles,
                                 TagID tag);

    Expected<> update_role(const protos::UpdateSubscriptionServiceRequest& update_req);

  private:
    void add_role(const std::string& name);
    Role& get_role(const std::string& name);

    void do_issue_update() final;
    void do_drop_tag(const TagID& tag) final;

    std::string m_name;

    // roles are defines at construction time in the body of the constructor
    // no new keys should be added
    std::map<std::string, std::unique_ptr<Role>> m_roles;
};

/**
 * @brief Component of SubscriptionService that holds state for each Role
 *
 * A Role has a set of members and subscribers. When either list is updated, the Role's nonce is incremented. When the
 * nonce is greater than the value of the nonce on last update, an update can be issued by calling issue_update.
 *
 * An issue_update will send a protos::SubscriptionServiceUpdate to all subscribers containing the (tag, instance_id)
 * tuple for each item in the members list.
 */
class Role final : public VersionedState
{
  public:
    Role(std::string service_name, std::string role_name) :
      m_service_name(std::move(service_name)),
      m_role_name(std::move(role_name))
    {}

    // subscribers are notified when new members are added
    void add_member(std::uint64_t tag, std::shared_ptr<server::ClientInstance> instance);
    void add_subscriber(std::uint64_t tag, std::shared_ptr<server::ClientInstance> instance);

    // drop a client instance - this will remove the instaces from both the
    // members and subscribers list
    void drop_tag(std::uint64_t tag);

    // subscribers will report when they have completed an update for a given nonce
    // this will enable us to fence on that update, meaning if we drop a tagged member resulting in the
    // server-side nonce being incremented to X, we can fence that value X with the subscribers so that at some point in
    // the future, all m_subscriber_nonces should be X or greater.
    void update_subscriber_nonce(const std::uint64_t& tag, const std::uint64_t& nonce);

    const std::string& service_name() const final;
    const std::string& role_name() const;

  private:
    bool has_update() const final;
    void do_make_update(protos::StateUpdate& update) const final;
    void do_issue_update(const protos::StateUpdate& update) final;

    // this method evaluates the state of the latched tags with respect to the state of the subscribers
    // once all subscribers are sufficiently up-to-date, latched tags can be dropped.
    void evaluate_latches();

    // enqueue the state update to be written to the client
    static void await_update(const std::shared_ptr<server::ClientInstance>& instance,
                             const protos::StateUpdate& update);

    std::string m_service_name;
    std::string m_role_name;

    // <tag, instance> - set of tagged instances will will be sent to subscribers on update
    std::map<std::uint64_t, std::shared_ptr<server::ClientInstance>> m_members;

    // <tag, instance> - set of tagged instances to receive updates of the member map
    std::map<std::uint64_t, std::shared_ptr<server::ClientInstance>> m_subscribers;

    // <tag, nonce> - holds the current nonce/state of the client - updated on update_subscriber_nonce
    std::map<std::uint64_t, std::uint64_t> m_subscriber_nonces;

    // <tag, <nonce, instance>> - when all m_subscriber_nonces are >= nonce issue drop event
    std::map<std::uint64_t, std::pair<std::uint64_t, std::shared_ptr<server::ClientInstance>>> m_latched_members;
};

}  // namespace mrc::internal::control_plane::server

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

#include "internal/control_plane/client/subscription_service.hpp"

#include "internal/control_plane/client.hpp"
#include "internal/control_plane/client/instance.hpp"
#include "internal/expected.hpp"
#include "internal/service.hpp"
#include "internal/utils/contains.hpp"

#include "mrc/protos/architect.pb.h"

#include <glog/logging.h>

#include <ostream>
#include <utility>

namespace mrc::internal::control_plane::client {

SubscriptionService::SubscriptionService(const std::string& service_name, Instance& instance) :
  m_service_name(std::move(service_name)),
  m_instance(instance)
{
    service_set_description(
        MRC_CONCAT_STR("subscription_service " << m_service_name << "[" << instance.instance_id() << "]"));
}

SubscriptionService::~SubscriptionService()
{
    Service::call_in_destructor();
}

const std::string& SubscriptionService::service_name() const
{
    return m_service_name;
}

RoleUpdater& SubscriptionService::subscriptions(const std::string& role)
{
    DCHECK(contains(m_subscriptions, role));
    return *m_subscriptions.at(role);
}

void SubscriptionService::request_stop()
{
    // issue a request to drop this subscription service
    // after confirmation form the control plane, teardown will be executed by the updater, which will formally
    // stop and join any outstanding runnables
    if (Service::state() == ServiceState::Running)
    {
        drop_subscription_service();
    }
}

void SubscriptionService::do_service_start()
{
    // register this instance with the control plane
    // if successful, this object now has a valid and globally unique tag
    register_subscription_service();

    // create update endpoints for the client instance to write updates
    for (const auto& role : subscribe_to_roles())
    {
        m_subscriptions[role] = std::make_unique<RoleUpdater>(*this, role);
    }

    // virtual method to enable derived class to specialize
    DVLOG(10) << "[start] setup subscription service: " << service_name() << ", role: " << role();
    do_subscription_service_setup();
    DVLOG(10) << "[success] setup subscription service: " << service_name() << ", role: " << role();

    // register this subscription service with the client instance
    // this connects this object to the update stream
    m_instance.register_subscription_service(this->shared_from_this());

    // inform the control plane that this object can now begin to receieve updates
    activate_subscription_service();
}

// a stop is a kill
void SubscriptionService::do_service_stop()
{
    do_service_kill();
}

// perform teardown
void SubscriptionService::do_service_kill()
{
    do_subscription_service_teardown();
}

// our start method is not async
void SubscriptionService::do_service_await_live() {}

//
void SubscriptionService::do_service_await_join()
{
    do_subscription_service_join();
}

void SubscriptionService::register_subscription_service()
{
    DVLOG(10) << "get/create subscription service: " << service_name();
    {
        protos::CreateSubscriptionServiceRequest req;
        req.set_service_name(service_name());
        for (const auto& role : roles())
        {
            req.add_roles(role);
        }
        auto resp =
            m_instance.client().await_unary<protos::Ack>(protos::ClientUnaryCreateSubscriptionService, std::move(req));
        MRC_THROW_ON_ERROR(resp);
    }
    DVLOG(10) << "subscribtion_service: " << service_name() << " is live on the control plane server";

    DVLOG(10) << "register subscription service: " << service_name() << "; role: " << role();
    {
        protos::RegisterSubscriptionServiceRequest req;
        req.set_instance_id(m_instance.instance_id());
        req.set_service_name(service_name());
        req.set_role(role());
        for (const auto& role : subscribe_to_roles())
        {
            req.add_subscribe_to_roles(role);
        }
        auto resp = m_instance.client().await_unary<protos::RegisterSubscriptionServiceResponse>(
            protos::ClientUnaryRegisterSubscriptionService, std::move(req));
        MRC_THROW_ON_ERROR(resp);
        m_tag = resp->tag();
    }
    DVLOG(10) << "registered subscription_service: " << service_name() << "; role: " << role() << "; tag: " << m_tag;
}

void SubscriptionService::activate_subscription_service()
{
    DVLOG(10) << "[start] activate subscription service: " << service_name() << "; role: " << role()
              << "; tag: " << tag();
    CHECK(tag() != 0);
    protos::ActivateSubscriptionServiceRequest req;
    req.set_instance_id(m_instance.instance_id());
    req.set_service_name(service_name());
    req.set_role(role());
    req.set_tag(tag());
    for (const auto& role : subscribe_to_roles())
    {
        req.add_subscribe_to_roles(role);
    }
    MRC_THROW_ON_ERROR(m_instance.client().template await_unary<protos::Ack>(
        protos::ClientUnaryActivateSubscriptionService, std::move(req)));

    DVLOG(10) << "[finish] activate subscription service: " << service_name() << "; role: " << role()
              << "; tag: " << tag();
}

void SubscriptionService::drop_subscription_service()
{
    DVLOG(10) << "[start] drop subscription service: " << service_name() << "; role: " << role() << "; tag: " << tag();
    CHECK(tag() != 0);
    protos::DropSubscriptionServiceRequest req;
    req.set_service_name(service_name());
    req.set_instance_id(m_instance.instance_id());
    req.set_tag(tag());
    auto resp =
        m_instance.client().await_unary<protos::Ack>(protos::ClientUnaryDropSubscriptionService, std::move(req));
    MRC_THROW_ON_ERROR(resp);
    DVLOG(10) << "[finish] drop subscription service: " << service_name() << "; role: " << role() << "; tag: " << tag();
}

const std::uint64_t& SubscriptionService::tag() const
{
    DCHECK(m_tag != 0);
    return m_tag;
}

RoleUpdater::RoleUpdater(SubscriptionService& subscription_service, std::string role_name) :
  m_subscription_service(subscription_service),
  m_role_name(std::move(role_name))
{
    DCHECK(contains(m_subscription_service.subscribe_to_roles(), m_role_name));
}

void SubscriptionService::await_join()
{
    service_await_join();
}
bool SubscriptionService::is_live() const
{
    auto state = Service::state();
    return (state == ServiceState::Running);
}

void SubscriptionService::teardown()
{
    service_stop();
}

const mrc::runnable::LaunchOptions& SubscriptionService::policy_engine_launch_options() const
{
    return m_instance.client().launch_options();
}

void SubscriptionService::await_start()
{
    service_start();
}
bool SubscriptionService::is_startable() const
{
    return is_service_startable();
}
}  // namespace mrc::internal::control_plane::client

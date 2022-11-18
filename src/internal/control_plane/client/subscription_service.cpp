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
#include "internal/utils/contains.hpp"

#include <glog/logging.h>

#include <algorithm>

namespace srf::internal::control_plane::client {

SubscriptionService::SubscriptionService(const std::string& service_name, Instance& instance) :
  m_service_name(std::move(service_name)),
  m_instance(instance)
{
    service_set_description(
        SRF_CONCAT_STR("subscription_service " << m_service_name << "[" << instance.instance_id() << "]"));
}

SubscriptionService::~SubscriptionService()
{
    Service::call_in_destructor();
}

const std::string& SubscriptionService::service_name() const
{
    return m_service_name;
}

Role& SubscriptionService::subscriptions(const std::string& role)
{
    DCHECK(contains(m_subscriptions, role));
    return *m_subscriptions.at(role);
}

void SubscriptionService::do_service_start()
{
    register_subscription_service();

    for (const auto& role : subscribe_to_roles())
    {
        m_subscriptions[role] = std::make_unique<Role>(*this, role);
    }
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
        SRF_THROW_ON_ERROR(resp);
    }
    DVLOG(10) << "subscribtion_service: " << service_name() << " is live on the control plane server";

    DVLOG(10) << "register subscription service: " << service_name() << "; role: " << role();
    {
        protos::RegisterSubscriptionServiceRequest req;
        req.set_instance_id(m_instance.instance_id());
        req.set_service_name(this->service_name());
        req.set_role(role());
        for (const auto& role : subscribe_to_roles())
        {
            req.add_subscribe_to_roles(role);
        }
        auto resp = m_instance.client().await_unary<protos::RegisterSubscriptionServiceResponse>(
            protos::ClientUnaryRegisterSubscriptionService, std::move(req));
        SRF_THROW_ON_ERROR(resp);
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
    req.set_service_name(this->service_name());
    req.set_role(role());
    req.set_tag(tag());
    for (const auto& role : subscribe_to_roles())
    {
        req.add_subscribe_to_roles(role);
    }
    SRF_THROW_ON_ERROR(m_instance.client().template await_unary<protos::Ack>(
        protos::ClientUnaryActivateSubscriptionService, std::move(req)));

    DVLOG(10) << "[finish] activate subscription service: " << service_name() << "; role: " << role()
              << "; tag: " << tag();
}

std::function<void()> SubscriptionService::drop_subscription_service() const
{
    auto service_name = this->service_name();
    auto role         = this->role();
    auto tag          = this->tag();
    auto& instance    = m_instance;

    // note we are capturing this as a lambda because the moment we issue the drop subscription request
    // this object becomes volatile. using the lambda, we don't have to capture this as part of the
    // handles custom deleter, instead, we only capture the lambda

    return [service_name, role, tag, &instance] {
        DVLOG(10) << "[start] drop subscription service: " << service_name << "; role: " << role << "; tag: " << tag;
        protos::DropSubscriptionServiceRequest req;
        req.set_service_name(service_name);
        req.set_instance_id(instance.instance_id());
        req.set_tag(tag);
        auto resp =
            instance.client().await_unary<protos::Ack>(protos::ClientUnaryDropSubscriptionService, std::move(req));
        LOG_IF(ERROR, !resp) << resp.error().message();
        DVLOG(10) << "[finish] drop subscription service: " << service_name << "; role: " << role << "; tag: " << tag;
    };
}

const std::uint64_t& SubscriptionService::tag() const
{
    DCHECK(m_tag != 0);
    return m_tag;
}

Role::Role(SubscriptionService& subscription_service, std::string role_name) :
  m_subscription_service(subscription_service),
  m_role_name(std::move(role_name))
{
    DCHECK(contains(m_subscription_service.subscribe_to_roles(), m_role_name));
}

}  // namespace srf::internal::control_plane::client

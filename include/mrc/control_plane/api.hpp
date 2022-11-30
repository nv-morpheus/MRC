/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <set>
#include <string>

namespace mrc::control_plane {

/**
 * @brief Public interface for control plane subscription services
 *
 * A subscription service on the control plane is defined by a globally unique service_name with a set of roles. An
 * instance of a subscriptions service is bound to a specific role and a list of roles for which updates will be
 * provided.
 *
 * Example, the MRC Publisher-Subscriber (pubsub) is a service with two roles {"Publisher", "Subscriber"}. An instance
 * of a Publisher object has the role() set to "Publisher" and gets a globally unqiue 64-bit tag/identifier. Publishers
 * subscribe to updates from the Subscriber role. When Subscribers are added or removed, the control plane will issue
 * updates that will be delivered to the Publisher object.
 *
 * This interface is broken down into two components, one focused on the identity and one focused the controls.
 */
struct ISubscriptionService;

struct ISubscriptionServiceIdentity
{
    virtual ~ISubscriptionServiceIdentity() = default;

    // name of subscription service
    virtual const std::string& service_name() const = 0;

    // globally unique tag for this instance
    virtual const std::uint64_t& tag() const = 0;

    // the specific role of this instance
    virtual const std::string& role() const = 0;

    // the set of roles for which this instance will receive updates
    virtual const std::set<std::string>& subscribe_to_roles() const = 0;
};

struct ISubscriptionServiceControls
{
    virtual ~ISubscriptionServiceControls() = default;

    // request that the subscription service be stopped
    virtual void request_stop() = 0;

    // bring up the service
    virtual void await_start() = 0;

    // await the completion of the subscription service
    virtual void await_join() = 0;

    // true/false if the subscription service is startable
    virtual bool is_startable() const = 0;

    // true/false if the subscription service is live
    virtual bool is_live() const = 0;
};

struct ISubscriptionService : public virtual ISubscriptionServiceIdentity, public virtual ISubscriptionServiceControls
{};

}  // namespace mrc::control_plane

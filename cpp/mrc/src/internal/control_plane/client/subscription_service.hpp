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

#include "mrc/control_plane/api.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/macros.hpp"

#include <boost/fiber/future/promise.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace mrc::internal::control_plane::client {

class Instance;
class RoleUpdater;

class ISubscriptionServiceUpdater : public virtual mrc::control_plane::ISubscriptionServiceIdentity
{
  public:
    ~ISubscriptionServiceUpdater() override = default;

    // informs the subscription service that is can teardown in internals because it is fully disconnected from both the
    // control plane and the data plane
    virtual void teardown() = 0;

    // access the role updater object for a given role - this is how control plane updates are applied
    virtual RoleUpdater& subscriptions(const std::string& role) = 0;

    friend Instance;
};

class SubscriptionService : public virtual mrc::control_plane::ISubscriptionService,
                            public ISubscriptionServiceUpdater,
                            public std::enable_shared_from_this<SubscriptionService>,
                            private Service
{
  public:
    SubscriptionService(const std::string& service_name, Instance& instance);
    ~SubscriptionService() override;

    // // todo(cpp20) - template<concepts::subscription_service T, typename... Args>
    // template <typename T, typename... ArgsT>
    // static auto create(ArgsT&&... args) -> std::shared_ptr<T>
    // {
    //     auto service = std::shared_ptr<T>(new T(std::forward<ArgsT>(args)...));
    //     return service;
    // }

    // [mrc::control_plane::ISubscriptionServiceIdentity] name of service
    const std::string& service_name() const final;

    // [mrc::control_plane::ISubscriptionServiceIdentity] globally unique tag for this instance
    const std::uint64_t& tag() const final;

    // the set of possible roles for this service
    virtual const std::set<std::string>& roles() const = 0;

    // [mrc::control_plane::ISubscriptionServiceControl] indicates that the public api has requested a stop
    // note: the subscription service is not stopped/joined until getting a drop_subscription_service event the server
    void request_stop() final;

    // [mrc::control_plane::ISubscriptionServiceControl] indicates that the public api has requested an await_start
    void await_start() final;

    // [mrc::control_plane::ISubscriptionServiceControl] indicates that the public api has requested an await_join
    void await_join() final;

    // [mrc::control_plane::ISubscriptionServiceControl]
    bool is_startable() const final;

    // [mrc::control_plane::ISubscriptionServiceControl]
    bool is_live() const final;

  protected:
    const mrc::runnable::LaunchOptions& policy_engine_launch_options() const;

  private:
    // this method is executed when the control plane client receives an update for this subscription service
    // the update from the server will container the role and map of tags to instances ids
    virtual void update_tagged_instances(const std::string& role,
                                         const std::unordered_map<std::uint64_t, InstanceID>& tagged_instances) = 0;

    // derived class must override this method
    virtual void do_subscription_service_setup()    = 0;
    virtual void do_subscription_service_teardown() = 0;
    virtual void do_subscription_service_join()     = 0;

    // [SubscriptionServiceUpdater] - teardown the subscription service
    void teardown() final;

    // [SubscriptionServiceUpdater] - can only be accessed after start
    RoleUpdater& subscriptions(const std::string& role) final;

    // [Service overrides]
    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;

    // registers the subscription service with the control plane; will not receive updates until activated
    void register_subscription_service();

    // activates the subscription service with the control plane enabling updates to begin
    void activate_subscription_service();

    //
    void drop_subscription_service();

    const std::string m_service_name;
    std::uint64_t m_tag{0};
    Instance& m_instance;
    std::map<std::string, std::unique_ptr<RoleUpdater>> m_subscriptions;

    friend RoleUpdater;
};

class RoleUpdater final
{
  public:
    RoleUpdater(SubscriptionService& subscription_service, std::string role_name);
    virtual ~RoleUpdater() = default;

    DELETE_COPYABILITY(RoleUpdater);
    DELETE_MOVEABILITY(RoleUpdater);

  private:
    void update_tagged_instances(const std::unordered_map<std::uint64_t, InstanceID>& tagged_instances)
    {
        m_subscription_service.update_tagged_instances(m_role_name, tagged_instances);
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        for (auto& p : m_update_promises)
        {
            p.set_value();
        }
        m_update_promises.clear();
    }

    SubscriptionService& m_subscription_service;
    const std::string m_role_name;
    std::vector<Promise<void>> m_update_promises;
    std::mutex m_mutex;

    friend Instance;
};

}  // namespace mrc::internal::control_plane::client

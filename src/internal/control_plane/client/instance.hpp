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
#include "internal/network/resources.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/service.hpp"

#include "mrc/node/source_channel.hpp"
#include "mrc/protos/architect.pb.h"
#include "mrc/runnable/runner.hpp"
#include "mrc/types.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace mrc::internal::control_plane::client {

class ISubscriptionServiceUpdater;

class Instance final : private resources::PartitionResourceBase, private Service
{
  public:
    Instance(Client& client,
             InstanceID instance_id,
             resources::PartitionResourceBase& base,
             mrc::node::SourceChannel<const protos::StateUpdate>& update_channel);
    ~Instance() final;

    Client& client();
    const InstanceID& instance_id() const;

    void register_subscription_service(std::shared_ptr<ISubscriptionServiceUpdater> service);

  private:
    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;

    Future<void> shutdown();

    void do_handle_state_update(const protos::StateUpdate& update);
    void do_update_subscription_state(const std::string& service_name,
                                      const std::uint64_t& nonce,
                                      const protos::UpdateSubscriptionServiceState& update);
    void do_drop_subscription_state(const std::string& service_name,
                                    const protos::DropSubscriptionServiceState& update);

    Client& m_client;
    const InstanceID m_instance_id;
    Promise<void> m_shutdown_promise;
    std::multimap<std::string, std::shared_ptr<ISubscriptionServiceUpdater>> m_subscription_services;
    std::unique_ptr<mrc::runnable::Runner> m_update_handler;

    friend network::Resources;
};

}  // namespace mrc::internal::control_plane::client

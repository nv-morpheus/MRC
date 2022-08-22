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

#include "internal/expected.hpp"
#include "internal/network/resources.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/ucx/common.hpp"

#include "srf/channel/status.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/protos/architect.pb.h"
#include "srf/runnable/runner.hpp"
#include "srf/types.hpp"

#include <set>
#include <string>

namespace srf::internal::control_plane {
class Client;
}

namespace srf::internal::data_plane {
class Client;
}

namespace srf::internal::control_plane::client {

class SubscriptionService;

class Instance final : private resources::PartitionResourceBase
{
  public:
    Instance(Client& client,
             InstanceID instance_id,
             resources::PartitionResourceBase& base,
             srf::node::SourceChannel<const protos::StateUpdate>& update_channel);
    ~Instance() final;

    Client& client();
    const InstanceID& instance_id() const;

    void register_subscription_service(std::unique_ptr<SubscriptionService> service);

  private:
    void do_handle_state_update(const protos::StateUpdate& update);
    void do_update_subscription_state(const std::string& service_name,
                                      const std::uint64_t& nonce,
                                      const protos::UpdateSubscriptionServiceState& update);
    void do_drop_subscription_state(const std::string& service_name,
                                    const protos::DropSubscriptionServiceState& update);

    Client& m_client;
    data_plane::Client* m_data_plane{nullptr};
    const InstanceID m_instance_id;
    std::unique_ptr<srf::runnable::Runner> m_update_handler;

    std::multimap<std::string, std::unique_ptr<SubscriptionService>> m_subscription_services;

    friend network::Resources;
};

}  // namespace srf::internal::control_plane::client

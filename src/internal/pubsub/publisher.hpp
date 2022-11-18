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

#include "internal/pubsub/pub_sub_base.hpp"
#include "internal/resources/forward.hpp"

#include "srf/node/source_channel.hpp"
#include "srf/pubsub/api.hpp"
#include "srf/utils/macros.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace srf::internal::pubsub {

class PublisherBackend;

class Publisher final : public PubSubBase, public srf::pubsub::IPublisher
{
    Publisher(std::string service_name, resources::PartitionResources& resources);

  public:
    ~Publisher() final = default;

    DELETE_COPYABILITY(Publisher);
    DELETE_MOVEABILITY(Publisher);

    // ISubscriptionService overrides

    void stop() final;
    bool is_live() const final;
    void await_join() final;

    // IPublisher overrides

    std::unique_ptr<srf::codable::ICodableStorage> create_storage() final;

  private:
    const std::string& role() const final
    {
        return role_publisher();
    }
    const std::set<std::string>& subscribe_to_roles() const final
    {
        static std::set<std::string> r = {role_subscriber()};
        return r;
    }

    virtual void on_update()                = 0;

    void update_tagged_instances(const std::string& role,
                                 const std::unordered_map<std::uint64_t, InstanceID>& tagged_instances) final
    {
        DCHECK_EQ(role, role_subscriber());

        // // todo - convert tagged instances -> tagged endpoints
        // m_tagged_instances = tagged_instances;
        // for (const auto& [tag, instance_id] : m_tagged_instances)
        // {
        //     // m_tagged_endpoints[tag] = resources().network()->data_plane().client().endpoint_shared(instance_id);
        // }
        on_update();
    }

    void do_service_await_live() override;
    void do_service_stop() override;
    void do_service_kill() override;
    void do_service_await_join() override;

    // resources - needs to be a PartitionRuntime
    resources::PartitionResources& m_resources;

    // set of tags

    friend PublisherBackend;
};

}  // namespace srf::internal::pubsub

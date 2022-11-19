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

#include "internal/pubsub/base.hpp"
#include "internal/resources/forward.hpp"
#include "internal/ucx/endpoint.hpp"

#include "srf/codable/forward.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/pubsub/api.hpp"
#include "srf/runtime/remote_descriptor.hpp"
#include "srf/utils/macros.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

namespace srf::internal::pubsub {

class Publisher : public Base,
                  public srf::pubsub::IPublisher,
                  public srf::node::SourceChannelWriteable<srf::runtime::RemoteDescriptor>
{
    Publisher(std::string service_name, runtime::Runtime& runtime);

  public:
    ~Publisher() override = default;

    DELETE_COPYABILITY(Publisher);
    DELETE_MOVEABILITY(Publisher);

    // [IPublisher] publish a remote descriptor
    channel::Status publish(srf::runtime::RemoteDescriptor&& rd) final;

    // [IPublisher] publish an encoded object
    channel::Status publish(std::unique_ptr<srf::codable::EncodedStorage> encoded_object) final;

    // [ISubscriptionServiceIdentity] provide the value for the role of this instance
    const std::string& role() const final;

    // [ISubscriptionServiceIdentity] provide the set of roles for which updates will be delivered
    const std::set<std::string>& subscribe_to_roles() const final;

  protected:
    // sends a remote descriptor to a remote endpoint over the data plane with a globally unique tag
    // note: the tag is required to differentiate multiple subscribers on the same endpoint
    void publish(srf::runtime::RemoteDescriptor&& rd,
                 const std::uint64_t& tag,
                 std::shared_ptr<ucx::Endpoint> endpoint);

    // current set of tagged instances
    const std::unordered_map<std::uint64_t, InstanceID>& tagged_instances() const;

    // current set of tagged endpoints
    const std::unordered_map<std::uint64_t, std::shared_ptr<ucx::Endpoint>>& tagged_endpoints() const;

  private:
    // [IPublisher] provides a runtime dependent codable storage object
    std::unique_ptr<srf::codable::ICodableStorage> create_storage() final;

    // [internal::control_plane::client::SubscriptionService]
    // setup up the runnables needed to driver the publisher
    void do_subscription_service_setup() final;

    // [internal::control_plane::client::SubscriptionService]
    // teardown up the runnables needed to driver the publisher
    void do_subscription_service_teardown() final;

    // [internal::control_plane::client::SubscriptionService]
    // await on the completion of all internal runnables
    void do_subscription_service_join() final;

    // [internal::control_plane::client::SubscriptionService]
    // called by the update engine when updates for a given subscribed_to role is received
    void update_tagged_instances(const std::string& role,
                                 const std::unordered_map<std::uint64_t, InstanceID>& tagged_instances) final;

    // apply policy
    virtual void apply_policy(srf::runtime::RemoteDescriptor&& rd) = 0;

    // called immediate on completion of update_tagged_instances
    virtual void on_update() = 0;

    // resources - needs to be a PartitionRuntime
    runtime::Runtime& m_runtime;

    // policy engine runner
    std::unique_ptr<srf::runnable::Runner> m_policy_engine;

    // set of active tagged instances for subscribers
    std::unordered_map<std::uint64_t, InstanceID> m_tagged_instances;

    // set of current tagged endpoints for subscribers
    std::unordered_map<std::uint64_t, std::shared_ptr<ucx::Endpoint>> m_tagged_endpoints;
};

}  // namespace srf::internal::pubsub

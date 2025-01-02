/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/pubsub/api.hpp"
#include "mrc/runtime/remote_descriptor.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/macros.hpp"

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

namespace mrc::runnable {
class Runner;
}  // namespace mrc::runnable
namespace mrc::memory {
class TransientBuffer;
}  // namespace mrc::memory
namespace mrc::runtime {
class Partition;
}  // namespace mrc::runtime

namespace mrc::pubsub {

/**
 * @brief The internal type-erased SubscriberService
 *
 */
class SubscriberService final : public Base, public mrc::pubsub::ISubscriberService
{
    SubscriberService(std::string service_name, runtime::Partition& runtime);

  public:
    ~SubscriberService() override;

    DELETE_COPYABILITY(SubscriberService);
    DELETE_MOVEABILITY(SubscriberService);

    // [ISubscriptionServiceIdentity] provide the value for the role of this instance
    const std::string& role() const final;

    // [ISubscriptionServiceIdentity] provide the set of roles for which updates will be delivered
    const std::set<std::string>& subscribe_to_roles() const final;

  private:
    // [control_plane::client::SubscriptionService]
    // setup up the runnables needed to driver the publisher
    void do_subscription_service_setup() final;

    // [control_plane::client::SubscriptionService]
    // teardown up the runnables needed to driver the publisher
    void do_subscription_service_teardown() final;

    // [control_plane::client::SubscriptionService]
    // await on the completion of all internal runnables
    void do_subscription_service_join() final;

    // [control_plane::client::SubscriptionService]
    // called by the update engine when updates for a given subscribed_to role is received
    void update_tagged_instances(const std::string& role,
                                 const std::unordered_map<std::uint64_t, InstanceID>& tagged_instances) final;

    // deserialize the incoming protobuf and create a local remote descriptor
    mrc::runtime::RemoteDescriptor network_handler(memory::TransientBuffer& buffer);

    // runner for the network handler node
    std::unique_ptr<mrc::runnable::Runner> m_network_handler;

    // limit access to the constructor; this object must be constructed as a shared_ptr
    friend runtime::Partition;
};

}  // namespace mrc::pubsub

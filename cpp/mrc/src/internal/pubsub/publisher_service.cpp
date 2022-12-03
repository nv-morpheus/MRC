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

#include "internal/pubsub/publisher_service.hpp"

#include "internal/codable/codable_storage.hpp"
#include "internal/data_plane/client.hpp"
#include "internal/data_plane/resources.hpp"
#include "internal/network/resources.hpp"
#include "internal/remote_descriptor/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/runnable/resources.hpp"

#include "mrc/core/utils.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runtime/remote_descriptor.hpp"

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <optional>
#include <utility>
#include <vector>

namespace mrc::internal::pubsub {

PublisherService::PublisherService(std::string service_name, runtime::Partition& runtime) :
  Base(std::move(service_name), runtime),
  m_runtime(runtime)
{}

channel::Status PublisherService::publish(mrc::runtime::RemoteDescriptor&& rd)
{
    return this->await_write(std::move(rd));
}

channel::Status PublisherService::publish(std::unique_ptr<mrc::codable::EncodedStorage> encoded_object)
{
    auto rd = m_runtime.remote_descriptor_manager().register_encoded_object(std::move(encoded_object));
    return this->await_write(std::move(rd));
}

std::unique_ptr<mrc::codable::ICodableStorage> PublisherService::create_storage()
{
    return std::make_unique<codable::CodableStorage>(m_runtime.resources());
}

const std::string& PublisherService::role() const
{
    return role_publisher();
}

const std::set<std::string>& PublisherService::subscribe_to_roles() const
{
    static std::set<std::string> r = {role_subscriber()};
    return r;
}
void PublisherService::update_tagged_instances(const std::string& role,
                                               const std::unordered_map<std::uint64_t, InstanceID>& tagged_instances)
{
    DCHECK_EQ(role, role_subscriber());

    auto cur_tags         = extract_keys(m_tagged_instances);
    auto new_tags         = extract_keys(tagged_instances);
    auto [added, removed] = set_compare(cur_tags, new_tags);

    for (const auto& tag : removed)
    {
        CHECK_EQ(m_tagged_endpoints.erase(tag), 1);
    }

    for (const auto& tag : added)
    {
        m_tagged_endpoints[tag] =
            resources().network()->data_plane().client().endpoint_shared(tagged_instances.at(tag));
    }

    m_tagged_instances = std::move(tagged_instances);

    // allow derived classes to take some action on_update
    on_update();
}

void PublisherService::do_subscription_service_setup()
{
    auto policy_engine = std::make_unique<mrc::node::RxSink<mrc::runtime::RemoteDescriptor>>(
        [this](mrc::runtime::RemoteDescriptor rd) { apply_policy(std::move(rd)); });

    // form an edge to this object's SourceChannelWritable
    mrc::node::make_edge(*this, *policy_engine);

    // launch the policy engine on the same fiber pool as the updater
    m_policy_engine = m_runtime.resources()
                          .runnable()
                          .launch_control()
                          .prepare_launcher(policy_engine_launch_options(), std::move(policy_engine))
                          ->ignition();

    m_policy_engine->await_live();
}

void PublisherService::do_subscription_service_teardown()
{
    release_channel();
}

void PublisherService::do_subscription_service_join()
{
    m_policy_engine->await_join();
}

void PublisherService::publish(mrc::runtime::RemoteDescriptor&& rd,
                               const std::uint64_t& tag,
                               std::shared_ptr<ucx::Endpoint> endpoint)
{
    // todo(cpp20) - bracket initializer
    // {.rd = std::move(rd), .endpoint = std::move(endpoint), .tag = tag}
    resources().network()->data_plane().client().remote_descriptor_channel().await_write(
        {std::move(rd), std::move(endpoint), tag});
}

const std::unordered_map<std::uint64_t, InstanceID>& PublisherService::tagged_instances() const
{
    return m_tagged_instances;
}
const std::unordered_map<std::uint64_t, std::shared_ptr<ucx::Endpoint>>& PublisherService::tagged_endpoints() const
{
    return m_tagged_endpoints;
}
}  // namespace mrc::internal::pubsub

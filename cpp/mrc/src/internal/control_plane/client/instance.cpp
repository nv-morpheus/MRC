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

#include "internal/control_plane/client/instance.hpp"

#include "internal/control_plane/client.hpp"
#include "internal/control_plane/client/subscription_service.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/utils/contains.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/protos/architect.pb.h"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/operations.hpp>
#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ostream>
#include <type_traits>  // IWYU pragma: keep
#include <unordered_map>
#include <utility>
#include <vector>

namespace mrc::internal::control_plane::client {

Instance::Instance(Client& client,
                   InstanceID instance_id,
                   resources::PartitionResourceBase& base,
                   mrc::node::SourceChannel<const protos::StateUpdate>& update_channel) :
  resources::PartitionResourceBase(base),
  m_client(client),
  m_instance_id(instance_id)
{
    auto update_handler = std::make_unique<mrc::node::RxSink<protos::StateUpdate>>(
        [this](protos::StateUpdate update) { do_handle_state_update(update); });

    mrc::node::make_edge(update_channel, *update_handler);

    m_update_handler =
        runnable().launch_control().prepare_launcher(client.launch_options(), std::move(update_handler))->ignition();

    service_start();
    service_await_live();
}

Instance::~Instance()
{
    service_kill();
    service_await_join();
    Service::call_in_destructor();
}

void Instance::do_service_start() {}
void Instance::do_service_stop()
{
    service_kill();
}
void Instance::do_service_kill()
{
    while (!m_subscription_services.empty())
    {
        boost::this_fiber::sleep_for(std::chrono::milliseconds(100));
    }

    DVLOG(10) << "client instance: " << m_instance_id << " issuing drop request";
    protos::TaggedInstance msg;
    msg.set_instance_id(m_instance_id);
    CHECK(client().await_unary<protos::Ack>(protos::ClientUnaryDropWorker, std::move(msg)));

    // requesting an update to avoid the timeout
    client().request_update();

    // this should block until an update is issued by the server for the client to finalize the instance drop
    m_update_handler->await_join();
    DVLOG(10) << "client instance: " << m_instance_id << " dropped by server - shutting down client-side";

    m_shutdown_promise.set_value();
}
void Instance::do_service_await_live()
{
    m_update_handler->await_live();
}
void Instance::do_service_await_join() {}

Future<void> Instance::shutdown()
{
    service_stop();
    return m_shutdown_promise.get_future();
}

void Instance::register_subscription_service(std::shared_ptr<ISubscriptionServiceUpdater> subscription_service)
{
    auto it = m_subscription_services.emplace(subscription_service->service_name(), std::move(subscription_service));
}

void Instance::do_handle_state_update(const protos::StateUpdate& update)
{
    if (update.has_update_subscription_service())
    {
        DVLOG(10) << "control plane instance on partition " << partition_id() << " got an update msg for "
                  << update.service_name();
        return do_update_subscription_state(
            update.service_name(), update.nonce(), update.update_subscription_service());
    }

    if (update.has_drop_subscription_service())
    {
        const auto& service = update.drop_subscription_service();

        DCHECK_GT(m_subscription_services.count(update.service_name()), 0)
            << "failed to find active subscription service with name: " << update.service_name();

        DVLOG(10) << "client::instance [" << partition_id() << "] dropping tag: " << service.tag()
                  << "; for role: " << service.role();

        auto range = m_subscription_services.equal_range(update.service_name());
        for (auto it = range.first; it != range.second;)
        {
            auto& service = *it->second;
            if (service.role() == service.role() && service.tag() == service.tag())
            {
                DVLOG(10) << "client dropping subscription service: " << update.service_name()
                          << "; role: " << service.role() << "; tag: " << service.tag();

                ISubscriptionServiceUpdater& updater = service;
                updater.teardown();
                it = m_subscription_services.erase(it);
            }
            else
            {
                it++;
            }
        }
    }
}

void Instance::do_update_subscription_state(const std::string& service_name,
                                            const std::uint64_t& nonce,
                                            const protos::UpdateSubscriptionServiceState& update)
{
    auto range = m_subscription_services.equal_range(service_name);
    std::vector<std::uint64_t> tags;
    std::unordered_map<std::uint64_t, InstanceID> tagged_instances;
    for (auto it = range.first; it != range.second; it++)
    {
        auto& service = *it->second;
        if (contains(service.subscribe_to_roles(), update.role()))
        {
            // lazy populate
            if (tagged_instances.empty())
            {
                for (const auto& ti : update.tagged_instances())
                {
                    tagged_instances[ti.tag()] = ti.instance_id();
                }
            }
            DVLOG(10) << "client::Instance[" << partition_id() << "]: updating service: " << service.service_name()
                      << "; role: " << service.role() << "; tag: " << service.tag() << "; with "
                      << tagged_instances.size() << " tagged instances";
            service.subscriptions(update.role()).update_tagged_instances(tagged_instances);
            tags.push_back(service.tag());
        }
    }
    if (!tags.empty())
    {
        protos::UpdateSubscriptionServiceRequest req;
        req.set_service_name(service_name);
        req.set_role(update.role());
        req.set_nonce(nonce);
        for (const auto& tag : tags)
        {
            req.add_tags(tag);
        }
        client().issue_event(protos::ClientEventUpdateSubscriptionService, std::move(req));
    }
}

const InstanceID& Instance::instance_id() const
{
    return m_instance_id;
}

Client& Instance::client()
{
    return m_client;
}

// void Instance::attach_data_plane_client(data_plane::Client* data_plane)
// {
//     CHECK(data_plane);
//     std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//     CHECK(m_data_plane == nullptr);
//     m_data_plane = data_plane;
//     for (const auto& [id, address] : m_worker_addresses)
//     {
//         m_data_plane->register_instance(id, address);
//     }
// }

}  // namespace mrc::internal::control_plane::client

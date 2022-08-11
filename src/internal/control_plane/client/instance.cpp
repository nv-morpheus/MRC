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
#include "internal/data_plane/client.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/utils/contains.hpp"

#include "srf/node/edge_builder.hpp"
#include "srf/protos/architect.pb.h"

namespace srf::internal::control_plane::client {

Instance::Instance(Client& client,
                   InstanceID instance_id,
                   resources::PartitionResourceBase& base,
                   srf::node::SourceChannel<protos::StateUpdate>& update_channel) :
  resources::PartitionResourceBase(base),
  m_client(client),
  m_instance_id(instance_id)
{
    auto update_handler = std::make_unique<srf::node::RxSink<protos::StateUpdate>>(
        [this](protos::StateUpdate update) { do_handle_state_update(update); });

    srf::node::make_edge(update_channel, *update_handler);

    m_update_handler =
        runnable().launch_control().prepare_launcher(client.launch_options(), std::move(update_handler))->ignition();
}

Instance::~Instance()
{
    m_client.drop_instance(m_instance_id);
    m_update_handler->await_join();
    DVLOG(10) << "client instance " << m_instance_id << " completed";
}

void Instance::do_handle_state_update(const protos::StateUpdate& update)
{
    DVLOG(10) << "control plane instance on partition " << partition_id() << " got an update for "
             << update.service_name();

    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    if (update.has_update())
    {
        if (update.update())
        {
            // true - starting update
            DCHECK(!m_update_in_progress);
            m_update_in_progress = true;
        }
        else
        {
            // false - finishing update
            DCHECK(m_update_in_progress);
            m_update_in_progress = false;
            for (auto& p : m_update_promises)
            {
                p.set_value();
            }
            m_update_promises.clear();
        }
    }
    else if (update.has_connections())
    {
        do_connections_update(update.connections());
    }
    else if (update.has_subscription_service())
    {
        // do subscription update
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

Future<void> Instance::fence_update()
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return m_update_promises.emplace_back().get_future();
}

void Instance::do_connections_update(const protos::ConnectionsState& connections)
{
    std::set<InstanceID> new_instance_ids;
    for (const auto& tagged_instance : connections.tagged_instances())
    {
        new_instance_ids.insert(tagged_instance.instance_id());
    }

    std::set<InstanceID> missing_worker_addresses;
    std::set_difference(new_instance_ids.begin(),
                        new_instance_ids.end(),
                        begin_keys(m_worker_addresses),
                        end_keys(m_worker_addresses),
                        std::inserter(missing_worker_addresses, missing_worker_addresses.end()));

    DVLOG(10) << "control_plane connection update; missing " << missing_worker_addresses.size() << " worker addresses";

    std::set<InstanceID> remove_instances;
    std::set_difference(begin_keys(m_worker_addresses),
                        end_keys(m_worker_addresses),
                        new_instance_ids.begin(),
                        new_instance_ids.end(),
                        std::inserter(remove_instances, remove_instances.end()));

    DVLOG(10) << "control_plane connection update; removing " << remove_instances.size() << " worker addresses";

    for (const auto& id : remove_instances)
    {
        m_worker_addresses.erase(id);
        if (m_data_plane != nullptr)
        {
            m_data_plane->drop_connection(id);
        }
    }

    if (!missing_worker_addresses.empty())
    {
        DVLOG(10) << "fetching worker addresses for " << missing_worker_addresses.size() << " instances";
        protos::LookupWorkersRequest req;
        for (const auto& id : missing_worker_addresses)
        {
            req.add_instance_ids(id);
        }

        auto resp = client().await_unary<protos::LookupWorkersResponse>(
            protos::EventType::ClientUnaryLookupWorkerAddresses, std::move(req));

        if (!resp)
        {
            LOG(ERROR) << "unary fetch of worker addresses failed: " << resp.error().message();
            return;
        }

        LOG(INFO) << "got back " << resp->worker_addresses_size() << " new worker addresses";
        for (const auto& worker : resp->worker_addresses())
        {
            DVLOG(10) << "registering ucx worker address for instance_id: " << worker.instance_id();
            m_worker_addresses[worker.instance_id()] = worker.worker_address();
            if (m_data_plane != nullptr)
            {
                m_data_plane->register_instance(worker.instance_id(), worker.worker_address());
            }
        }
    }
}
void Instance::attach_data_plane_client(data_plane::Client* data_plane)
{
    CHECK(data_plane);
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    CHECK(m_data_plane == nullptr);
    m_data_plane = data_plane;
    for (const auto& [id, address] : m_worker_addresses)
    {
        m_data_plane->register_instance(id, address);
    }
}

std::size_t Instance::ucx_worker_address_count() const
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return m_worker_addresses.size();
}
}  // namespace srf::internal::control_plane::client

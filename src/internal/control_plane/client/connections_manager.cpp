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

#include "internal/control_plane/client/connections_manager.hpp"

#include "internal/control_plane/client.hpp"
#include "internal/control_plane/client/instance.hpp"
#include "internal/expected.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/ucx/resources.hpp"
#include "internal/ucx/worker.hpp"
#include "internal/utils/contains.hpp"

#include "mrc/core/task_queue.hpp"
#include "mrc/protos/architect.pb.h"

#include <ext/alloc_traits.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <ostream>
#include <set>
#include <string>
#include <type_traits>  // IWYU pragma: keep
#include <utility>

namespace mrc::internal::control_plane::client {

ConnectionsManager::ConnectionsManager(Client& client, update_channel_t& update_channel) : StateManager(client)
{
    this->start_with_channel(update_channel);
}

ConnectionsManager::~ConnectionsManager()
{
    this->await_join();
}

const std::vector<InstanceID>& ConnectionsManager::instance_ids() const
{
    DCHECK(client().state() == Client::State::Operational);
    return m_instance_ids;
}

std::map<InstanceID, std::unique_ptr<client::Instance>> ConnectionsManager::register_ucx_addresses(
    std::vector<std::optional<ucx::Resources>>& ucx_resources)
{
    CHECK(client().state() == Client::State::RegisteringWorkers);

    protos::RegisterWorkersRequest req;
    for (auto& ucx : ucx_resources)
    {
        DCHECK(ucx);
        req.add_ucx_worker_addresses(ucx->worker().address());
    }

    auto resp =
        client().await_unary<protos::RegisterWorkersResponse>(protos::ClientUnaryRegisterWorkers, std::move(req));
    CHECK_EQ(resp->instance_ids_size(), ucx_resources.size());

    m_machine_id = resp->machine_id();
    std::map<InstanceID, std::unique_ptr<client::Instance>> instances;

    for (int i = 0; i < resp->instance_ids_size(); i++)
    {
        auto id = resp->instance_ids().at(i);
        m_instance_ids.push_back(id);
        m_worker_addresses[id] = ucx_resources.at(i)->worker().address();
        m_update_channels[id]  = std::make_unique<update_channel_t>();
        instances[id] =
            std::make_unique<client::Instance>(client(), id, *ucx_resources.at(i), *m_update_channels.at(id));
    }

    // issue activate event - connection events from the server will
    client().await_unary<protos::Ack>(protos::ClientUnaryActivateStream, std::move(*resp));

    DVLOG(10) << "client - machine_id: " << m_machine_id;
    return instances;
}

void ConnectionsManager::do_update(const protos::StateUpdate&& update_msg)
{
    DCHECK(client().runnable().main().caller_on_same_thread());

    if (update_msg.has_connections())
    {
        return do_connections_update(update_msg.connections());
    }

    LOG(FATAL) << "unhandled update";
}

void ConnectionsManager::do_connections_update(const protos::UpdateConnectionsState& connections)
{
    std::set<InstanceID> new_instance_ids;
    for (const auto& tagged_instance : connections.tagged_instances())
    {
        new_instance_ids.insert(tagged_instance.instance_id());
    }

    DVLOG(10) << "after update the client will have " << new_instance_ids.size() << " connections";

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

        // this will drop the instance and allow the client::Instance to complete destruction
        m_update_channels.erase(id);
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

        DVLOG(10) << "got back " << resp->worker_addresses_size() << " new worker addresses";
        for (const auto& worker : resp->worker_addresses())
        {
            DVLOG(10) << "registering ucx worker address for instance_id: " << worker.instance_id();
            m_worker_addresses[worker.instance_id()] = worker.worker_address();
        }
    }
}

const std::map<InstanceID, MachineID>& ConnectionsManager::locality_map() const
{
    DCHECK(client().runnable().main().caller_on_same_thread());
    return m_locality_map;
}

const std::map<InstanceID, ucx::WorkerAddress>& ConnectionsManager::worker_addresses() const
{
    // DCHECK(client().runnable().main().caller_on_same_thread());
    return m_worker_addresses;
}

const std::map<InstanceID, std::unique_ptr<ConnectionsManager::update_channel_t>>&
ConnectionsManager::instance_channels() const
{
    DCHECK(client().runnable().main().caller_on_same_thread());
    return m_update_channels;
}

}  // namespace mrc::internal::control_plane::client

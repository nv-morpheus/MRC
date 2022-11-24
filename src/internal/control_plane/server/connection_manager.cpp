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

#include "internal/control_plane/server/connection_manager.hpp"

#include "internal/utils/contains.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/protos/architect.pb.h"

#include <glog/logging.h>
#include <google/protobuf/any.pb.h>

#include <algorithm>
#include <ostream>
#include <utility>

namespace mrc::internal::control_plane::server {

void ConnectionManager::add_stream(const stream_t& stream)
{
    auto search = m_streams.find(stream->get_id());
    if (search != m_streams.end())  // todo(cpp20) - unlikely
    {
        LOG(FATAL) << "non-unique stream registration detected";
    }
    m_streams[stream->get_id()] = stream;
}

void ConnectionManager::drop_stream(const stream_id_t& stream_id) noexcept
{
    DVLOG(10) << "dropping stream_id: " << stream_id;

    auto stream = m_streams.find(stream_id);
    if (stream == m_streams.end())
    {
        DLOG(FATAL) << "stream_id: " << stream_id << " does not exist";
        return;
    }

    // drop each client instance associated with stream_id
    auto range = m_instances_by_stream.equal_range(stream_id);
    for (auto& i = range.first; i != range.second; i++)
    {
        DVLOG(10) << "dropping instance_id: " << i->second;
        DCHECK(contains(m_instances, i->second));
        m_instances.erase(i->second);
    }
    m_instances_by_stream.erase(stream_id);

    // issue finish and await the stream
    auto writer = stream->second->writer();
    if (writer)
    {
        writer->finish();
        writer.reset();
    }
    stream->second->await_fini();

    // finally drop the drop stream from the map
    m_streams.erase(stream_id);
    mark_as_modified();
}

void ConnectionManager::drop_all_streams() noexcept
{
    std::vector<stream_id_t> ids;
    for (auto& [stream_id, stream] : m_streams)
    {
        ids.push_back(stream_id);
    }
    for (const auto& id : ids)
    {
        drop_stream(id);
    }
}

Expected<ConnectionManager::instance_t> ConnectionManager::get_instance(const instance_id_t& instance_id) const
{
    auto search = m_instances.find(instance_id);
    if (search == m_instances.end())
    {
        return Error::create(MRC_CONCAT_STR("unable to acquire instance with id: " << instance_id));
    }
    return search->second;
}

std::vector<ConnectionManager::instance_id_t> ConnectionManager::get_instance_ids(const stream_id_t& stream_id) const
{
    std::vector<instance_id_t> ids;
    auto range = m_instances_by_stream.equal_range(stream_id);
    for (auto& i = range.first; i != range.second; i++)
    {
        DCHECK_EQ(i->first, stream_id);
        ids.push_back(i->second);
    }
    return ids;
}

Expected<protos::RegisterWorkersResponse> ConnectionManager::register_instances(
    const writer_t& writer, const protos::RegisterWorkersRequest& req)
{
    const auto stream_id = writer->get_id();

    // validate that the worker addresses are valid before updating state
    for (const auto& worker_address : req.ucx_worker_addresses())
    {
        if (contains(m_ucx_worker_addresses, worker_address))  // todo(cpp20) - contains
        {
            return Error::create("invalid ucx worker address(es) - duplicate registration(s) detected");
        }
    }

    // check if any workers/instances have been registered on the requesting stream
    if (m_instances_by_stream.count(stream_id) != 0)
    {
        return Error::create(MRC_CONCAT_STR("failed to register instances on immutable stream "
                                            << stream_id << "; streams are immutable after first registration"));
    }

    // set machine id for the current stream
    protos::RegisterWorkersResponse response;
    response.set_machine_id(stream_id);

    for (const auto& worker_address : req.ucx_worker_addresses())
    {
        // create server-side client instances which hold the worker address and stream writer
        auto instance = std::make_shared<server::ClientInstance>(writer, worker_address);

        if (contains(m_instances, instance->get_id()))  // todo(cpp20) contains and unlikely
        {
            throw Error::create(ErrorCode::Fatal, "non-unique instance_id detected");
        }

        DVLOG(10) << "registered instance_id: " << instance->get_id() << " on stream_id: " << stream_id;
        m_ucx_worker_addresses.insert(worker_address);
        m_instances[instance->get_id()] = instance;
        response.add_instance_ids(instance->get_id());
    }

    // mark worked as updated
    mark_as_modified();

    return response;
}

Expected<protos::Ack> ConnectionManager::drop_instance(const writer_t& writer, const protos::TaggedInstance& req)
{
    const auto stream_id = writer->get_id();
    auto instance        = get_instance(req.instance_id());
    MRC_EXPECT(instance);

    DCHECK(contains(m_ucx_worker_addresses, instance.value()->worker_address()));
    m_ucx_worker_addresses.erase(instance.value()->worker_address());

    auto range = m_instances_by_stream.equal_range(stream_id);
    for (auto i = range.first; i != range.second; i++)
    {
        if (i->second == req.instance_id())
        {
            m_instances_by_stream.erase(i);
            break;
        }
    }

    m_instances.erase(req.instance_id());

    mark_as_modified();

    DVLOG(10) << "instance " << req.instance_id() << " dropped";
    return protos::Ack{};
}

Expected<protos::Ack> ConnectionManager::activate_stream(const writer_t& writer,
                                                         const protos::RegisterWorkersResponse& message)
{
    auto stream_id = writer->get_id();
    for (const auto& instance_id : message.instance_ids())
    {
        m_instances_by_stream.insert(std::pair{stream_id, instance_id});
    }
    return {};
}

Expected<protos::LookupWorkersResponse> ConnectionManager::lookup_workers(const writer_t& writer,
                                                                          const protos::LookupWorkersRequest& req) const
{
    protos::LookupWorkersResponse resp;
    for (const auto& id : req.instance_ids())
    {
        auto instance = get_instance(id);
        if (instance)
        {
            auto* worker = resp.add_worker_addresses();
            worker->set_instance_id(id);
            worker->set_machine_id(instance.value()->stream_writer().get_id());
            worker->set_worker_address(instance.value()->worker_address());
        }
        else
        {
            LOG(WARNING) << "unable to lookup instance_id: " << id;
        }
    }
    return resp;
}

bool ConnectionManager::has_update() const
{
    return true;
}

void ConnectionManager::do_make_update(protos::StateUpdate& update) const
{
    auto* connections = update.mutable_connections();
    for (const auto& [machine_id, instance_id] : m_instances_by_stream)
    {
        auto* msg = connections->add_tagged_instances();
        msg->set_instance_id(instance_id);
        msg->set_tag(machine_id);
    }
}

void ConnectionManager::do_issue_update(const protos::StateUpdate& update)
{
    protos::Event event;
    event.set_event(protos::EventType::ServerStateUpdate);
    event.set_tag(0);  // explicit broadcast to all partitions
    event.mutable_message()->PackFrom(update);

    for (const auto& [stream_id, stream] : m_streams)
    {
        auto writer = stream->writer();
        if (writer)
        {
            auto status = writer->await_write(event);
            LOG_IF(WARNING, status != channel::Status::success)
                << "failed to issue connections update to stream/machine_id: " << stream_id;
        }
    }
}

const std::string& ConnectionManager::service_name() const
{
    static std::string name = "connection_manager";
    return name;
}

const std::map<ConnectionManager::stream_id_t, ConnectionManager::stream_t>& ConnectionManager::streams() const
{
    return m_streams;
}

}  // namespace mrc::internal::control_plane::server

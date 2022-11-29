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

#include "internal/control_plane/server/client_instance.hpp"
#include "internal/control_plane/server/versioned_issuer.hpp"
#include "internal/expected.hpp"
#include "internal/grpc/server_streaming.hpp"
#include "internal/grpc/stream_writer.hpp"

#include "mrc/protos/architect.pb.h"
#include "mrc/types.hpp"

#include <cstddef>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace mrc::internal::control_plane::server {

/**
 * @brief Control Plane Connection Manager
 *
 * Manages each gRPC bidirectional stream via the mrc::internal::rpc::ServerStream connection.
 *
 * Each stream/connection is allowed a one-time registration of client instances (client-side partitions) to be
 * associated with the stream.
 *
 * The ConnectionManager is a VersionedState. When requested, ConnectionManager will issue a single
 * protos::ServerUpdate to all connected stream with the protos::Event::tag() value set to 0, which means that update
 * message will be broadcast to all partition subscribers on the client.
 *
 * The ServerUpdate event will be composed of a protos::ServerUpdateConnections which is a list of TaggedInstances,
 * where the tag is the stream_id (unique machine_id).
 *
 * The protos::ServerUpdateConnections will not send the UCX worker addresses. It is up to the client to determine the
 * set of UCX worker addresses missing from its local registar and issue an unary rpc to fetch worker addresses request
 * to the control plane.
 *
 * @note This object is not thread-safe. It is assumed the owner of this object will properly control exclusive access.
 */
class ConnectionManager : public VersionedState
{
  public:
    using stream_t      = std::shared_ptr<rpc::ServerStream<mrc::protos::Event, mrc::protos::Event>>;
    using writer_t      = std::shared_ptr<rpc::StreamWriter<mrc::protos::Event>>;
    using event_t       = stream_t::element_type::IncomingData;
    using instance_t    = std::shared_ptr<server::ClientInstance>;
    using stream_id_t   = std::size_t;
    using instance_id_t = std::size_t;

    void add_stream(const stream_t& stream);
    void drop_stream(const stream_id_t& stream_id) noexcept;
    void drop_all_streams() noexcept;

    const std::map<stream_id_t, stream_t>& streams() const;

    Expected<instance_t> get_instance(const instance_id_t& instance_id) const;

    std::vector<instance_id_t> get_instance_ids(const stream_id_t& stream_id) const;

    Expected<protos::RegisterWorkersResponse> register_instances(const writer_t& writer,
                                                                 const protos::RegisterWorkersRequest& req);

    Expected<protos::LookupWorkersResponse> lookup_workers(const writer_t& writer,
                                                           const protos::LookupWorkersRequest& req) const;

    Expected<protos::Ack> activate_stream(const writer_t& writer, const protos::RegisterWorkersResponse& message);

    Expected<protos::Ack> drop_instance(const writer_t& writer, const protos::TaggedInstance& req);

    const std::string& service_name() const final;

  protected:
  private:
    bool has_update() const final;
    void do_make_update(protos::StateUpdate& update) const final;
    void do_issue_update(const protos::StateUpdate& update) final;

    MachineID m_machine_id;
    std::vector<InstanceID> m_instance_ids;

    // populated on registration
    std::map<stream_id_t, stream_t> m_streams;
    std::map<instance_id_t, instance_t> m_instances;
    std::set<std::string> m_ucx_worker_addresses;

    // populated on activation - updates issued from this map
    std::multimap<stream_id_t, instance_id_t> m_instances_by_stream;
};

}  // namespace mrc::internal::control_plane::server

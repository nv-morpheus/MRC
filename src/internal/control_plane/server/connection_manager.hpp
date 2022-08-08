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

#include "srf/protos/architect.pb.h"

#include <cstdint>
#include <map>
#include <set>

namespace srf::internal::control_plane::server {

class ConnectionManager : public VersionedIssuer
{
  public:
    using stream_t      = std::shared_ptr<rpc::ServerStream<srf::protos::Event, srf::protos::Event>>;
    using writer_t      = std::shared_ptr<rpc::StreamWriter<srf::protos::Event>>;
    using event_t       = stream_t::element_type::IncomingData;
    using instance_t    = std::shared_ptr<server::ClientInstance>;
    using stream_id_t   = std::size_t;
    using instance_id_t = std::size_t;

    void add_stream(const stream_t& stream);
    void drop_stream(const stream_id_t& stream_id) noexcept;
    void drop_all_streams() noexcept;

    Expected<instance_t> get_instance(const instance_id_t& instance_id) const;

    std::vector<instance_id_t> get_instance_ids(const stream_id_t& stream_id) const;

    Expected<protos::RegisterWorkersResponse> register_instances(const writer_t& writer,
                                                                 const protos::RegisterWorkersRequest& req);

    // Expected<protos::FetchWorkerAddressesResponse> fetch_worker_addresses(
    //     const protos::protos::FetchWorkerAddressesRequest& req) const;

  protected:
  private:
    const std::string& service_name() const final;
    void do_make_update(protos::ServiceUpdate& update) const final;
    void do_issue_update(const protos::ServiceUpdate& update) final;

    std::map<stream_id_t, stream_t> m_streams;
    std::map<instance_id_t, instance_t> m_instances;
    std::multimap<stream_id_t, instance_id_t> m_instances_by_stream;
    std::set<std::string> m_ucx_worker_addresses;
};

}  // namespace srf::internal::control_plane::server

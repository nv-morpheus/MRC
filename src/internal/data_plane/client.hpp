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

#include "internal/control_plane/client/connections_manager.hpp"
#include "internal/control_plane/client/state_manager.hpp"
#include "internal/data_plane/request.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/service.hpp"
#include "internal/ucx/common.hpp"
#include "internal/ucx/context.hpp"
#include "internal/ucx/endpoint.hpp"
#include "internal/ucx/worker.hpp"

#include "srf/channel/status.hpp"
#include "srf/codable/encoded_object.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/protos/remote_descriptor.pb.h"
#include "srf/runnable/launch_control.hpp"
#include "srf/runnable/runner.hpp"
#include "srf/types.hpp"

#include <rxcpp/rx.hpp>  // IWYU pragma: keep
#include <ucp/api/ucp_def.h>

#include <cstddef>
#include <map>
#include <memory>

namespace srf::internal::data_plane {

class Client final : public resources::PartitionResourceBase
{
  public:
    Client(resources::PartitionResourceBase& base,
           ucx::Resources& ucx,
           control_plane::client::ConnectionsManager& connections_manager);
    ~Client() final;

    std::shared_ptr<ucx::Endpoint> endpoint_shared(const InstanceID& instance_id) const;

    // drop endpoint
    void drop_endpoint(const InstanceID& instance_id);

    // number of established remote instances
    std::size_t endpoint_count() const;

    void async_p2p_recv(void* addr, std::size_t bytes, std::uint64_t tag, Request& request);
    void async_p2p_send(
        void* addr, std::size_t bytes, std::uint64_t tag, InstanceID instance_id, Request& request) const;

    void async_get(void* addr,
                   std::size_t bytes,
                   InstanceID instance_id,
                   void* remote_addr,
                   const std::string& packed_remote_key,
                   Request& request) const;

    static void async_recv(void* addr,
                           std::size_t bytes,
                           std::uint64_t tag,
                           std::uint64_t mask,
                           const ucx::Worker& worker,
                           Request& request);
    static void async_send(
        void* addr, std::size_t bytes, std::uint64_t tag, const ucx::Endpoint& endpoint, Request& request);

    const ucx::Endpoint& endpoint(const InstanceID& instance_id) const;

  private:
    ucx::Resources& m_ucx;
    control_plane::client::ConnectionsManager& m_connnection_manager;
    mutable std::map<InstanceID, std::shared_ptr<ucx::Endpoint>> m_endpoints;
};

}  // namespace srf::internal::data_plane

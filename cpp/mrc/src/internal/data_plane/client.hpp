/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/resources/partition_resources_base.hpp"
#include "internal/service.hpp"
#include "internal/ucx/worker.hpp"

#include "mrc/node/operators/node_component.hpp"
#include "mrc/runtime/remote_descriptor_handle.hpp"
#include "mrc/types.hpp"

#include <ucp/api/ucp_def.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>

namespace mrc::node {
template <typename T>
class WritableProvider;
}  // namespace mrc::node
namespace mrc::control_plane::client {
class ConnectionsManager;
}  // namespace mrc::control_plane::client
namespace mrc::memory {
class TransientPool;
}  // namespace mrc::memory
namespace mrc::ucx {
class Endpoint;
class UcxResources;
}  // namespace mrc::ucx
namespace mrc::runnable {
class Runner;
}  // namespace mrc::runnable

namespace mrc::data_plane {
class Request;
class DataPlaneResources;

struct LocalDescriptorMessage
{
    LocalDescriptorMessage() = default;
    LocalDescriptorMessage(runtime::LocalDescriptorHandle handle,
                           std::shared_ptr<ucx::Endpoint> endpoint,
                           std::uint64_t tag) :
      handle(std::move(handle)),
      endpoint(std::move(endpoint)),
      tag(tag)
    {}

    runtime::LocalDescriptorHandle handle;
    std::shared_ptr<ucx::Endpoint> endpoint;
    std::uint64_t tag;
};

class Client final : public resources::PartitionResourceBase, private Service
{
  public:
    Client(resources::PartitionResourceBase& base,
           ucx::UcxResources& ucx,
           control_plane::client::ConnectionsManager& connections_manager,
           memory::TransientPool& transient_pool);
    ~Client() final;

    std::shared_ptr<ucx::Endpoint> endpoint_shared(const InstanceID& instance_id) const;

    // drop endpoint
    void drop_endpoint(const InstanceID& instance_id);

    // number of established remote instances
    std::size_t endpoint_count() const;

    void async_p2p_recv(void* addr, std::size_t bytes, std::uint64_t tag, Request& request);
    void async_p2p_send(void* addr,
                        std::size_t bytes,
                        std::uint64_t tag,
                        InstanceID instance_id,
                        Request& request) const;

    node::WritableProvider<LocalDescriptorMessage>& remote_descriptor_channel();

    // primitive rdma and send/recv call

    static void async_get(void* addr,
                          std::size_t bytes,
                          const ucx::Endpoint& ep,
                          std::uint64_t remote_addr,
                          ucp_rkey_h rkey,
                          Request& request);

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
    static void async_send(void* addr,
                           std::size_t bytes,
                           std::uint64_t tag,
                           const ucx::Endpoint& endpoint,
                           Request& request);

    static void async_am_send(std::uint32_t id,
                              const void* header,
                              std::size_t header_length,
                              const ucx::Endpoint& endpoint,
                              Request& request);

    const ucx::Endpoint& endpoint(const InstanceID& instance_id) const;

  private:
    void issue_remote_descriptor(LocalDescriptorMessage msg);

    void do_service_start() final;
    void do_service_await_live() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_join() final;

    ucx::UcxResources& m_ucx;
    control_plane::client::ConnectionsManager& m_connnection_manager;
    memory::TransientPool& m_transient_pool;
    mutable std::map<InstanceID, std::shared_ptr<ucx::Endpoint>> m_endpoints;

    std::unique_ptr<mrc::runnable::Runner> m_rd_writer;
    std::unique_ptr<node::NodeComponent<LocalDescriptorMessage>> m_rd_channel;

    friend DataPlaneResources;
};

}  // namespace mrc::data_plane

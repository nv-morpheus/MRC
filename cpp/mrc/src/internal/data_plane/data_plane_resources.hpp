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

#include "internal/data_plane/request.hpp"
#include "internal/memory/transient_pool.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/service.hpp"
#include "internal/ucx/forward.hpp"

#include "mrc/memory/buffer_view.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/types.hpp"

#include <ucs/memory/memory_type.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ucxx {
class Context;
class Endpoint;
class Worker;
class Request;
class Address;
}  // namespace ucxx

namespace mrc::control_plane {
class Client;
}  // namespace mrc::control_plane
namespace mrc::memory {
class HostResources;
}  // namespace mrc::memory
namespace mrc::network {
class NetworkResources;
}  // namespace mrc::network
namespace mrc::ucx {
class RegistrationCache;
class RegistrationCache2;
class UcxResources;
}  // namespace mrc::ucx

namespace mrc::data_plane {
class Client;
class Server;

/**
 * @brief ArchitectResources hold and is responsible for constructing any object that depending the UCX data plane
 *
 */
class DataPlaneResources final : private Service, private resources::PartitionResourceBase
{
  public:
    DataPlaneResources(resources::PartitionResourceBase& base,
                       ucx::UcxResources& ucx,
                       memory::HostResources& host,
                       const InstanceID& instance_id,
                       control_plane::Client& control_plane_client);
    ~DataPlaneResources() final;

    Client& client();
    Server& server();

    const InstanceID& instance_id() const;
    std::string ucx_address() const;
    const ucx::RegistrationCache& registration_cache() const;

    static mrc::runnable::LaunchOptions launch_options(std::size_t concurrency);

  private:
    void do_service_start() final;
    void do_service_await_live() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_join() final;

    ucx::UcxResources& m_ucx;
    memory::HostResources& m_host;
    control_plane::Client& m_control_plane_client;
    InstanceID m_instance_id;

    memory::TransientPool m_transient_pool;
    std::unique_ptr<Server> m_server;
    std::unique_ptr<Client> m_client;

    friend network::NetworkResources;
};

class DataPlaneResources2
{
  public:
    DataPlaneResources2();
    ~DataPlaneResources2();

    ucxx::Context& context() const;

    ucxx::Worker& worker() const;

    std::string address() const;

    ucx::RegistrationCache2& registration_cache() const;

    std::shared_ptr<ucxx::Endpoint> create_endpoint(const std::string& address);

    // Advances the worker
    bool progress();

    // Flushes the worker
    bool flush();

    std::shared_ptr<ucxx::Request> tagged_send_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                     memory::const_buffer_view buffer_view,
                                                     uint64_t tag);

    std::shared_ptr<ucxx::Request> tagged_send_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                     const void* buffer,
                                                     size_t bytes,
                                                     uint64_t tag);

    std::shared_ptr<ucxx::Request> tagged_recv_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                     void* buffer,
                                                     size_t bytes,
                                                     uint64_t tag,
                                                     uint64_t tag_mask);

    std::shared_ptr<ucxx::Request> am_send_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                 memory::const_buffer_view buffer_view);

    std::shared_ptr<ucxx::Request> am_send_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                 const void* addr,
                                                 std::size_t bytes,
                                                 ucs_memory_type_t mem_type);
    std::shared_ptr<ucxx::Request> am_recv_async(std::shared_ptr<ucxx::Endpoint> endpoint);

  private:
    std::shared_ptr<ucxx::Context> m_context;
    std::shared_ptr<ucxx::Worker> m_worker;
    std::shared_ptr<ucxx::Address> m_address;

    std::shared_ptr<ucx::RegistrationCache2> m_registration_cache;

    std::map<std::string, std::shared_ptr<ucxx::Endpoint>> m_endpoints;
};

}  // namespace mrc::data_plane

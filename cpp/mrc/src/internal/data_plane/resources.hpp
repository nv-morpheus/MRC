/**
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

#include "internal/memory/transient_pool.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/service.hpp"

#include "mrc/runnable/launch_options.hpp"
#include "mrc/types.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace mrc::internal::control_plane {
class Client;
}  // namespace mrc::internal::control_plane
namespace mrc::internal::memory {
class HostResources;
}  // namespace mrc::internal::memory
namespace mrc::internal::network {
class NetworkResources;
}  // namespace mrc::internal::network
namespace mrc::internal::ucx {
class RegistrationCache;
class UcxResources;
}  // namespace mrc::internal::ucx

namespace mrc::internal::data_plane {
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

}  // namespace mrc::internal::data_plane

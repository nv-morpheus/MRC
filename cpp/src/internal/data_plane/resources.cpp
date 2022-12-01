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

#include "internal/data_plane/resources.hpp"

#include "internal/control_plane/client.hpp"
#include "internal/data_plane/client.hpp"
#include "internal/data_plane/server.hpp"
#include "internal/ucx/resources.hpp"
#include "internal/ucx/worker.hpp"

#include "mrc/memory/literals.hpp"

namespace mrc::internal::data_plane {

using namespace mrc::memory::literals;

Resources::Resources(resources::PartitionResourceBase& base,
                     ucx::Resources& ucx,
                     memory::HostResources& host,
                     const InstanceID& instance_id,
                     control_plane::Client& control_plane_client) :
  resources::PartitionResourceBase(base),
  m_ucx(ucx),
  m_host(host),
  m_control_plane_client(control_plane_client),
  m_instance_id(instance_id),
  m_transient_pool(32_MiB, 4, m_host.registered_memory_resource()),
  m_server(base, ucx, host, m_transient_pool, m_instance_id),
  m_client(base, ucx, m_control_plane_client.connections(), m_transient_pool)
{
    // ensure the data plane progress engine is up and running
    service_start();
    service_await_live();
}

Resources::~Resources()
{
    Service::call_in_destructor();
}

Client& Resources::client()
{
    return m_client;
}

// Server& Resources::server()
// {
//     return m_server;
// }

std::string Resources::ucx_address() const
{
    return m_ucx.worker().address();
}

const ucx::RegistrationCache& Resources::registration_cache() const
{
    return m_ucx.registration_cache();
}

void Resources::do_service_start()
{
    m_server.service_start();
    m_client.service_start();
}

void Resources::do_service_await_live()
{
    m_server.service_await_live();
    m_client.service_await_live();
}

void Resources::do_service_stop()
{
    // we only issue
    m_client.service_stop();
}

void Resources::do_service_kill()
{
    m_server.service_kill();
    m_client.service_kill();
}

void Resources::do_service_await_join()
{
    m_client.service_await_join();
    m_server.service_stop();
    m_server.service_await_join();
}

Server& Resources::server()
{
    return m_server;
}

mrc::runnable::LaunchOptions Resources::launch_options(std::size_t concurrency)
{
    return ucx::Resources::launch_options(concurrency);
}

const InstanceID& Resources::instance_id() const
{
    return m_instance_id;
}
}  // namespace mrc::internal::data_plane

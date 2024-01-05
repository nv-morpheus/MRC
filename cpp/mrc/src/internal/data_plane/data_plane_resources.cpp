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

#include "internal/data_plane/data_plane_resources.hpp"

#include "internal/data_plane/callbacks.hpp"
#include "internal/data_plane/client.hpp"
#include "internal/data_plane/server.hpp"
#include "internal/memory/host_resources.hpp"
#include "internal/ucx/endpoint.hpp"
#include "internal/ucx/ucx_resources.hpp"
#include "internal/ucx/utils.hpp"
#include "internal/ucx/worker.hpp"

#include "mrc/memory/literals.hpp"

#include <ucp/api/ucp.h>
#include <ucs/memory/memory_type.h>
#include <ucxx/api.h>

#include <memory>

namespace mrc::data_plane {

using namespace mrc::memory::literals;

DataPlaneResources::DataPlaneResources(resources::PartitionResourceBase& base,
                                       ucx::UcxResources& ucx,
                                       memory::HostResources& host,
                                       const InstanceID& instance_id,
                                       control_plane::Client& control_plane_client) :
  resources::PartitionResourceBase(base),
  Service("DataPlaneResources"),
  m_ucx(ucx),
  m_host(host),
  m_control_plane_client(control_plane_client),
  m_instance_id(instance_id),
  m_transient_pool(32_MiB, 4, m_host.registered_memory_resource()),
  m_server(std::make_unique<Server>(base, ucx, host, m_transient_pool, m_instance_id))
//   m_client(std::make_unique<Client>(base, ucx, m_control_plane_client.connections(), m_transient_pool))
{
    // ensure the data plane progress engine is up and running
    service_start();
    service_await_live();
}

DataPlaneResources::~DataPlaneResources()
{
    Service::call_in_destructor();
}

Client& DataPlaneResources::client()
{
    return *m_client;
}

std::string DataPlaneResources::ucx_address() const
{
    return m_ucx.worker().address();
}

const ucx::RegistrationCache& DataPlaneResources::registration_cache() const
{
    return m_ucx.registration_cache();
}

void DataPlaneResources::do_service_start()
{
    m_server->service_start();
    m_client->service_start();
}

void DataPlaneResources::do_service_await_live()
{
    m_server->service_await_live();
    m_client->service_await_live();
}

void DataPlaneResources::do_service_stop()
{
    // we only issue
    m_client->service_stop();
}

void DataPlaneResources::do_service_kill()
{
    m_server->service_kill();
    m_client->service_kill();
}

void DataPlaneResources::do_service_await_join()
{
    m_client->service_await_join();
    m_server->service_stop();
    m_server->service_await_join();
}

Server& DataPlaneResources::server()
{
    return *m_server;
}

mrc::runnable::LaunchOptions DataPlaneResources::launch_options(std::size_t concurrency)
{
    return ucx::UcxResources::launch_options(concurrency);
}

const InstanceID& DataPlaneResources::instance_id() const
{
    return m_instance_id;
}

DataPlaneResources2::DataPlaneResources2()
{
    DVLOG(10) << "initializing ucx context";

    int64_t featureFlags = UCP_FEATURE_TAG | UCP_FEATURE_AM | UCP_FEATURE_RMA;

    m_context = ucxx::createContext({}, featureFlags);

    DVLOG(10) << "initialize a ucx data_plane worker";
    m_worker = ucxx::createWorker(m_context, false, false);

    m_address = m_worker->getAddress();

    DVLOG(10) << "initialize the registration cache for this context";
    m_registration_cache = std::make_shared<ucx::RegistrationCache2>(m_context);

    // flush any work that needs to be done by the workers
    this->flush();
}

DataPlaneResources2::~DataPlaneResources2() {}

ucxx::Context& DataPlaneResources2::context() const
{
    return *m_context;
}

ucxx::Worker& DataPlaneResources2::worker() const
{
    return *m_worker;
}

std::string DataPlaneResources2::address() const
{
    return m_address->getString();
}

ucx::RegistrationCache2& DataPlaneResources2::registration_cache() const
{
    return *m_registration_cache;
}

std::shared_ptr<ucxx::Endpoint> DataPlaneResources2::create_endpoint(const ucx::WorkerAddress& address)
{
    auto address_obj = ucxx::createAddressFromString(address);

    auto endpoint = m_worker->createEndpointFromWorkerAddress(address_obj);

    m_endpoints[address] = endpoint;

    return endpoint;
}

bool DataPlaneResources2::progress()
{
    // Forward the worker once
    return m_worker->progressOnce();
}

bool DataPlaneResources2::flush()
{
    return m_worker->progress();
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::tagged_send_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                                      memory::const_buffer_view buffer_view,
                                                                      uint64_t tag)
{
    return this->tagged_send_async(endpoint, buffer_view.data(), buffer_view.bytes(), tag);
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::tagged_send_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                                      const void* buffer,
                                                                      size_t length,
                                                                      uint64_t tag)
{
    // TODO(MDD): Check that this EP belongs to this resource

    // Const cast away because UCXX only accepts void*
    auto request = endpoint->tagSend(const_cast<void*>(buffer), length, tag);

    return request;
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::tagged_recv_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                                      void* buffer,
                                                                      size_t length,
                                                                      uint64_t tag,
                                                                      uint64_t tag_mask)
{
    // TODO(MDD): Check that this EP belongs to this resource
    // TODO(MDD): Once 0.35 is released, support tag_mask
    auto request = endpoint->tagRecv(buffer, length, tag);

    return request;
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::am_send_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                                  memory::const_buffer_view buffer_view)
{
    return this->am_send_async(endpoint,
                               buffer_view.data(),
                               buffer_view.bytes(),
                               ucx::to_ucs_memory_type(buffer_view.kind()));
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::am_send_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                                  const void* addr,
                                                                  std::size_t bytes,
                                                                  ucs_memory_type_t mem_type)
{
    // TODO(MDD): Check that this EP belongs to this resource

    // Const cast away because UCXX only accepts void*
    auto request = endpoint->amSend(const_cast<void*>(addr), bytes, mem_type);

    return request;
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::am_recv_async(std::shared_ptr<ucxx::Endpoint> endpoint)
{
    // TODO(MDD): Check that this EP belongs to this resource
    auto request = endpoint->amRecv();

    return request;
}

// std::shared_ptr<ucxx::Request> DataPlaneResources2::receive_async2(void* addr,
//                                                             std::size_t bytes,
//                                                             std::uint64_t tag,
//                                                             std::uint64_t mask)
// {

//     ucxx::Endpoint endpoint(m_worker, m_worker->address());

//     auto request = endpoint.amRecv();

//     request.

//     return request;
// }

}  // namespace mrc::data_plane

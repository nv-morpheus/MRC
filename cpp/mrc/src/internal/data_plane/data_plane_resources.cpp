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
#include "internal/remote_descriptor/messages.hpp"
#include "internal/ucx/endpoint.hpp"
#include "internal/ucx/ucx_resources.hpp"
#include "internal/ucx/utils.hpp"
#include "internal/ucx/worker.hpp"

#include "mrc/coroutines/sync_wait.hpp"
#include "mrc/memory/literals.hpp"
#include "mrc/memory/resources/host/malloc_memory_resource.hpp"

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

void DataPlaneResources2::set_instance_id(uint64_t instance_id)
{
    m_instance_id = instance_id;
}

bool DataPlaneResources2::has_instance_id() const
{
    return m_instance_id.has_value();
}

uint64_t DataPlaneResources2::get_instance_id() const
{
    if (!this->has_instance_id())
    {
        throw std::runtime_error("Instance ID not set");
    }

    return m_instance_id.value();
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
    m_registration_cache3 = std::make_shared<ucx::RegistrationCache3>(m_context);

    auto pull_complete_callback = ucxx::AmReceiverCallbackType([this](std::shared_ptr<ucxx::Request> req) {
        auto status = req->getStatus();
        if (status != UCS_OK)
        {
            LOG(ERROR) << "Error calling pull_complete_callback, request failed with status " << status << "("
                       << ucs_status_string(status) << ")";
        }

        auto* message = reinterpret_cast<remote_descriptor::DescriptorPullCompletionMessage*>(
            req->getRecvBuffer()->data());

        complete_remote_pull(message);
    });
    m_worker->registerAmReceiverCallback(
        ucxx::AmReceiverCallbackInfo(ucxx::AmReceiverCallbackOwnerType("MRC"), ucxx::AmReceiverCallbackIdType(0)),
        pull_complete_callback);

    auto recv_process_message = ucxx::AmReceiverCallbackType([this](std::shared_ptr<ucxx::Request> req) {
        auto status = req->getStatus();
        if (status != UCS_OK)
        {
            LOG(ERROR) << "Error calling recv_process_message, request failed with status " << status << "("
                       << ucs_status_string(status) << ")";
        }

        // Create a descriptor from the received data
        auto buffer_view = memory::buffer_view(req->getRecvBuffer()->data(),
                                               req->getRecvBuffer()->getSize(),
                                               mrc::memory::memory_kind::host);

        std::shared_ptr<runtime::Descriptor2> recv_descriptor = runtime::Descriptor2::create_from_bytes(std::move(buffer_view), *this);

        // Although ClosableRingBuffer::write is a coroutine, write always completes instantaneously without awaiting.
        // ClosablRingBuffer size is always >= m_max_remote_descriptors, so there is always an empty slot.
        auto write_descriptor = [this, recv_descriptor]() -> coroutines::Task<void> {
            co_await m_recv_descriptors.write(recv_descriptor);
            co_return;
        };

        coroutines::sync_wait(write_descriptor());
    });
    m_worker->registerAmReceiverCallback(
        ucxx::AmReceiverCallbackInfo(ucxx::AmReceiverCallbackOwnerType("MRC"), ucxx::AmReceiverCallbackIdType(1)),
        recv_process_message);

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

ucx::RegistrationCache3& DataPlaneResources2::registration_cache3() const
{
    return *m_registration_cache3;
}

std::shared_ptr<ucxx::Endpoint> DataPlaneResources2::create_endpoint(const ucx::WorkerAddress& address,
                                                                     uint64_t instance_id)
{
    auto address_obj = ucxx::createAddressFromString(address);

    auto endpoint = m_worker->createEndpointFromWorkerAddress(address_obj);

    m_endpoints_by_address[address] = endpoint;
    m_endpoints_by_id[instance_id]  = endpoint;

    return endpoint;
}

bool DataPlaneResources2::has_endpoint(const std::string& address) const
{
    return m_endpoints_by_address.contains(address);
}

std::shared_ptr<ucxx::Endpoint> DataPlaneResources2::find_endpoint(const std::string& address) const
{
    return m_endpoints_by_address.at(address);
}

std::shared_ptr<ucxx::Endpoint> DataPlaneResources2::find_endpoint(uint64_t instance_id) const
{
    return m_endpoints_by_id.at(instance_id);
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

void DataPlaneResources2::wait_requests(const std::vector<std::shared_ptr<ucxx::Request>>& requests)
{
    auto remainingRequests = requests;
    while (!remainingRequests.empty())
    {
        auto updatedRequests = std::exchange(remainingRequests, decltype(remainingRequests)());
        for (auto const& r : updatedRequests)
        {
            this->progress();

            if (!r->isCompleted())
            {
                remainingRequests.push_back(r);
            }
            else
            {
                r->checkError();
            }
        }
    }
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::memory_send_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                                      memory::const_buffer_view buffer_view,
                                                                      uintptr_t remote_addr,
                                                                      ucp_rkey_h rkey)
{
    return this->memory_send_async(endpoint, buffer_view.data(), buffer_view.bytes(), remote_addr, rkey);
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::memory_send_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                                      const void* addr,
                                                                      std::size_t bytes,
                                                                      uintptr_t remote_addr,
                                                                      ucp_rkey_h rkey)
{
    // Const cast away because UCXX only accepts void*
    auto request = endpoint->memPut(const_cast<void*>(addr), bytes, remote_addr, rkey);

    return request;
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::memory_recv_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                                      memory::buffer_view buffer_view,
                                                                      uintptr_t remote_addr,
                                                                      const std::string& serialized_rkey)
{
    return this->memory_recv_async(endpoint, buffer_view.data(), buffer_view.bytes(), remote_addr, serialized_rkey);
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::memory_recv_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                                      void* addr,
                                                                      std::size_t bytes,
                                                                      uintptr_t remote_addr,
                                                                      const std::string& serialized_rkey)
{
    // Const cast away because UCXX only accepts void*
    auto rkey = ucxx::createRemoteKeyFromSerialized(endpoint, serialized_rkey);
    auto request = endpoint->memGet(addr, bytes, rkey);

    return request;
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
    auto request = endpoint->tagSend(const_cast<void*>(buffer), length, ucxx::Tag(tag));

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
    auto request = endpoint->tagRecv(buffer, length, ucxx::Tag(tag), ucxx::TagMaskFull);

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
    auto request = endpoint->amSend(const_cast<void*>(addr), bytes, mem_type, ucxx::AmReceiverCallbackInfo("MRC", 1));

    return request;
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::am_recv_async(std::shared_ptr<ucxx::Endpoint> endpoint)
{
    // TODO(MDD): Check that this EP belongs to this resource
    auto request = endpoint->amRecv();

    return request;
}

uint64_t DataPlaneResources2::get_next_object_id()
{
    return m_next_object_id++;
}

coroutines::Task<std::shared_ptr<runtime::Descriptor2>> DataPlaneResources2::await_recv_descriptor()
{
    auto read_element = co_await m_recv_descriptors.read();
    std::shared_ptr<runtime::Descriptor2> recv_descriptor = std::move(*read_element);

    recv_descriptor->fetch_remote_payloads();

    co_return recv_descriptor;
}

uint64_t DataPlaneResources2::register_remote_descriptor(std::shared_ptr<runtime::Descriptor2> descriptor)
{
    // If the descriptor has an object_id > 0, the descriptor has already been registered and should not be re-registered
    auto object_id = descriptor->encoded_object().object_id();
    if (object_id > 0)
    {
        m_descriptor_by_id[object_id].push_back(descriptor);
        return object_id;
    }

    object_id = get_next_object_id();
    descriptor->encoded_object().set_object_id(object_id);
    {
        std::unique_lock lock(m_remote_descriptors_mutex);
        m_remote_descriptors_cv.wait(lock, [this] {
            return m_descriptor_by_id.size() < m_max_remote_descriptors;
        });
        m_descriptor_by_id[object_id].push_back(descriptor);
    }
    return object_id;
}

uint64_t DataPlaneResources2::registered_remote_descriptor_count()
{
    return m_descriptor_by_id.size();
}

uint64_t DataPlaneResources2::registered_remote_descriptor_ptr_count(uint64_t object_id)
{
    return m_descriptor_by_id.at(object_id).size();
}

void DataPlaneResources2::complete_remote_pull(remote_descriptor::DescriptorPullCompletionMessage* message)
{
    // If the mapping between object_id to descriptor shared ptrs exists, then there exists >= 1 shared ptrs
    if (m_descriptor_by_id.find(message->object_id) != m_descriptor_by_id.end())
    {
        // Once we've completed pulling of a descriptor, we remove a descriptor shared ptr from the vector
        // When the vector becomes empty, there will be no more shared ptrs pointing to the descriptor object,
        // it will be destructed accordingly.
        // We should also remove that mapping as the object_id corresponding to that mapping will not be reused.
        auto& descriptors = m_descriptor_by_id[message->object_id];
        descriptors.pop_back();
        if (descriptors.size() == 0)
        {
            {
                std::unique_lock lock(m_remote_descriptors_mutex);
                m_descriptor_by_id.erase(message->object_id);
                m_registration_cache3->remove_descriptor(message->object_id);
            }
            m_remote_descriptors_cv.notify_one();
        }
    }
}

void DataPlaneResources2::set_max_remote_descriptors(uint64_t max_remote_descriptors)
{
    m_max_remote_descriptors = max_remote_descriptors;
    m_remote_descriptors_cv.notify_all();
}

}  // namespace mrc::data_plane

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

#include "mrc/channel/buffered_channel.hpp"
#include "mrc/memory/literals.hpp"

#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/insert_linebreaks.hpp>
#include <boost/archive/iterators/remove_whitespace.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <glog/logging.h>
#include <ucp/api/ucp.h>
#include <ucs/memory/memory_type.h>
#include <ucxx/api.h>

#include <iostream>
#include <memory>
#include <mutex>
#include <string>

namespace mrc::data_plane {

using namespace mrc::memory::literals;

std::string decode64(const std::string& val)
{
    using namespace boost::archive::iterators;
    using namespace std;

    typedef transform_width<binary_from_base64<remove_whitespace<string::const_iterator>>, 8, 6> it_binary_t;
    typedef insert_linebreaks<base64_from_binary<transform_width<string::const_iterator, 6, 8>>, 72> it_base64_t;

    std::string tmp = val;

    unsigned int paddChars = count(tmp.begin(), tmp.end(), '=');
    std::replace(tmp.begin(), tmp.end(), '=', 'A');                   // replace '=' by base64 encoding of '\0'
    string result(it_binary_t(tmp.begin()), it_binary_t(tmp.end()));  // decode
    result.erase(result.end() - paddChars, result.end());             // erase padding '\0' characters

    return result;

    // using It = transform_width<binary_from_base64<std::string::const_iterator>, 8, 6>;
    // return boost::algorithm::trim_right_copy_if(std::string(It(std::begin(val)), It(std::end(val))), [](char c) {
    //     return c == '\0';
    // });
}

std::string encode64(const std::string& val)
{
    using namespace boost::archive::iterators;
    using namespace std;

    typedef transform_width<binary_from_base64<remove_whitespace<string::const_iterator>>, 8, 6> it_binary_t;
    typedef insert_linebreaks<base64_from_binary<transform_width<string::const_iterator, 6, 8>>, 72> it_base64_t;

    // Encode
    unsigned int writePaddChars = (3 - val.length() % 3) % 3;
    string base64(it_base64_t(val.begin()), it_base64_t(val.end()));
    base64.append(writePaddChars, '=');

    return base64;

    // using It = base64_from_binary<transform_width<std::string::const_iterator, 6, 8>>;
    // auto tmp = std::string(It(std::begin(val)), It(std::end(val)));
    // return tmp.append((3 - val.size() % 3) % 3, '=');
}

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

DataPlaneResources2::DataPlaneResources2() :
  m_inbound_channel(std::make_unique<channel::BufferedChannel<std::unique_ptr<runtime::RemoteDescriptor2>>>())
{
    DVLOG(10) << "initializing ucx context";

    int64_t featureFlags = UCP_FEATURE_TAG | UCP_FEATURE_AM | UCP_FEATURE_RMA;

    m_context = ucxx::createContext({}, featureFlags);

    DVLOG(10) << "initialize a ucx data_plane worker";
    m_worker = ucxx::createWorker(m_context, false, false);

    m_address = m_worker->getAddress();

    auto test_str = m_address->getString();

    auto encoded = encode64(test_str);
    auto decoded = decode64(encoded);

    DCHECK_EQ(test_str, decoded) << "UCX Address Roundtrip failed";

    DVLOG(10) << "Created worker with address: " << this->address();

    DVLOG(10) << "initialize the registration cache for this context";
    m_registration_cache = std::make_shared<ucx::RegistrationCache2>(m_context);

    auto decrement_callback = ucxx::AmReceiverCallbackType([this](std::shared_ptr<ucxx::Request> req) {
        if (req->getStatus() != UCS_OK)
        {
            // TODO(Peter): Ensure the error gets raised somehow
            LOG(ERROR) << "Error calling decrement_callback";
        }

        auto* dec_message = reinterpret_cast<remote_descriptor::RemoteDescriptorDecrementMessage*>(
            req->getRecvBuffer()->data());

        if (dec_message->tokens > 0)
        {
            // Lock on usages of the descriptor map
            std::unique_lock lock(m_mutex);

            if (m_remote_descriptor_by_id.find(dec_message->object_id) == m_remote_descriptor_by_id.end())
            {
                LOG(ERROR) << "DataPlaneResources2[" << this->get_instance_id() << "]: Decrementing RD("
                           << dec_message->object_id << ") " << dec_message->tokens << ". Not found.";
                return;
            }

            auto remote_descriptor = m_remote_descriptor_by_id[dec_message->object_id];
            auto start_tokens      = remote_descriptor->encoded_object().tokens();
            auto remaining_tokens  = start_tokens - dec_message->tokens;

            remote_descriptor->encoded_object().set_tokens(remaining_tokens);

            if (remaining_tokens == 0)
            {
                // VLOG(10) << "DataPlaneResources2[" << this->get_instance_id() << "]: Decrementing RD("
                //          << dec_message->object_id << ") " << start_tokens << " -> " << dec_message->tokens << " -> "
                //          << remaining_tokens << ". Destroying.";

                m_remote_descriptor_by_id.erase(dec_message->object_id);
            }
            else
            {
                // VLOG(10) << "DataPlaneResources2[" << this->get_instance_id() << "]: Decrementing RD("
                //          << dec_message->object_id << ") " << start_tokens << " -> " << dec_message->tokens << " -> "
                //          << remaining_tokens << ".";
            }
        }
    });
    m_worker->registerAmReceiverCallback(
        ucxx::AmReceiverCallbackInfo(ucxx::AmReceiverCallbackOwnerType("MRC"), ucxx::AmReceiverCallbackIdType(0)),
        decrement_callback);

    auto remote_descriptor_callback = ucxx::AmReceiverCallbackType([this](std::shared_ptr<ucxx::Request> req) {
        if (req->getStatus() != UCS_OK)
        {
            LOG(ERROR) << "Error calling remote_descriptor_callback";
        }

        // Deserialize the remote descriptor
        auto recv_remote_descriptor = runtime::RemoteDescriptor2::from_bytes(
            {req->getRecvBuffer()->data(), req->getRecvBuffer()->getSize(), mrc::memory::memory_kind::host});

        // Write it to the inbound channel to be processed
        m_inbound_channel->await_write(std::move(recv_remote_descriptor));
    });
    m_worker->registerAmReceiverCallback(
        ucxx::AmReceiverCallbackInfo(ucxx::AmReceiverCallbackOwnerType("MRC"), ucxx::AmReceiverCallbackIdType(1 << 2)),
        remote_descriptor_callback);

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
    return encode64(m_address->getString());
}

ucx::RegistrationCache2& DataPlaneResources2::registration_cache() const
{
    return *m_registration_cache;
}

std::shared_ptr<ucxx::Endpoint> DataPlaneResources2::create_endpoint(const ucx::WorkerAddress& address,
                                                                     uint64_t instance_id)
{
    std::string decoded = decode64(address);

    auto address_obj = ucxx::createAddressFromString(decoded);

    auto endpoint = m_worker->createEndpointFromWorkerAddress(address_obj);

    m_endpoints_by_address[address] = endpoint;
    m_endpoints_by_id[instance_id]  = endpoint;

    DVLOG(10) << "Created endpoint with address: " << address;

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

std::shared_ptr<ucxx::Request> DataPlaneResources2::memory_send_async(
    std::shared_ptr<ucxx::Endpoint> endpoint,
    memory::const_buffer_view buffer_view,
    uintptr_t remote_addr,
    ucp_rkey_h rkey,
    ucxx::RequestCallbackUserFunction callback_function,
    ucxx::RequestCallbackUserData callback_data)
{
    return this->memory_send_async(endpoint,
                                   buffer_view.data(),
                                   buffer_view.bytes(),
                                   remote_addr,
                                   rkey,
                                   std::move(callback_function),
                                   std::move(callback_data));
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::memory_send_async(
    std::shared_ptr<ucxx::Endpoint> endpoint,
    const void* addr,
    std::size_t bytes,
    uintptr_t remote_addr,
    ucp_rkey_h rkey,
    ucxx::RequestCallbackUserFunction callback_function,
    ucxx::RequestCallbackUserData callback_data)
{
    // Const cast away because UCXX only accepts void*
    auto request = endpoint->memPut(const_cast<void*>(addr),
                                    bytes,
                                    remote_addr,
                                    rkey,
                                    false,
                                    std::move(callback_function),
                                    std::move(callback_data));

    return request;
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::memory_recv_async(
    std::shared_ptr<ucxx::Endpoint> endpoint,
    memory::buffer_view buffer_view,
    uintptr_t remote_addr,
    const void* packed_rkey_data,
    ucxx::RequestCallbackUserFunction callback_function,
    ucxx::RequestCallbackUserData callback_data)
{
    return this->memory_recv_async(endpoint,
                                   buffer_view.data(),
                                   buffer_view.bytes(),
                                   remote_addr,
                                   packed_rkey_data,
                                   std::move(callback_function),
                                   std::move(callback_data));
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::memory_recv_async(
    std::shared_ptr<ucxx::Endpoint> endpoint,
    void* addr,
    std::size_t bytes,
    uintptr_t remote_addr,
    const void* packed_rkey_data,
    ucxx::RequestCallbackUserFunction callback_function,
    ucxx::RequestCallbackUserData callback_data)
{
    ucp_rkey_h rkey;

    // Unpack the key
    auto rc = ucp_ep_rkey_unpack(endpoint->getHandle(), packed_rkey_data, &rkey);
    CHECK_EQ(rc, UCS_OK);

    // Const cast away because UCXX only accepts void*
    auto request = endpoint->memGet(
        addr,
        bytes,
        remote_addr,
        rkey,
        false,
        [rkey, callback_function](ucs_status_t status, std::shared_ptr<void> user_data) {
            ucp_rkey_destroy(rkey);

            if (callback_function)
            {
                callback_function(status, std::move(user_data));
            }
        },
        std::move(callback_data));

    return request;
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::tagged_send_async(
    std::shared_ptr<ucxx::Endpoint> endpoint,
    memory::const_buffer_view buffer_view,
    uint64_t tag,
    ucxx::RequestCallbackUserFunction callback_function,
    ucxx::RequestCallbackUserData callback_data)
{
    return this->tagged_send_async(endpoint,
                                   buffer_view.data(),
                                   buffer_view.bytes(),
                                   tag,
                                   std::move(callback_function),
                                   std::move(callback_data));
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::tagged_send_async(
    std::shared_ptr<ucxx::Endpoint> endpoint,
    const void* buffer,
    size_t length,
    uint64_t tag,
    ucxx::RequestCallbackUserFunction callback_function,
    ucxx::RequestCallbackUserData callback_data)
{
    // TODO(MDD): Check that this EP belongs to this resource

    // Const cast away because UCXX only accepts void*
    auto request = endpoint->tagSend(const_cast<void*>(buffer),
                                     length,
                                     ucxx::Tag(tag),
                                     false,
                                     std::move(callback_function),
                                     std::move(callback_data));

    return request;
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::tagged_recv_async(
    std::shared_ptr<ucxx::Endpoint> endpoint,
    void* buffer,
    size_t length,
    uint64_t tag,
    uint64_t tag_mask,
    ucxx::RequestCallbackUserFunction callback_function,
    ucxx::RequestCallbackUserData callback_data)
{
    // TODO(MDD): Check that this EP belongs to this resource
    // TODO(MDD): Once 0.35 is released, support tag_mask
    auto request = endpoint->tagRecv(buffer,
                                     length,
                                     ucxx::Tag(tag),
                                     ucxx::TagMaskFull,
                                     false,
                                     std::move(callback_function),
                                     std::move(callback_data));

    return request;
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::am_send_async(
    std::shared_ptr<ucxx::Endpoint> endpoint,
    memory::const_buffer_view buffer_view,
    std::optional<ucxx::AmReceiverCallbackInfo> callback_info,
    ucxx::RequestCallbackUserFunction callback_function,
    ucxx::RequestCallbackUserData callback_data)
{
    return this->am_send_async(endpoint,
                               buffer_view.data(),
                               buffer_view.bytes(),
                               ucx::to_ucs_memory_type(buffer_view.kind()),
                               callback_info,
                               std::move(callback_function),
                               std::move(callback_data));
}

std::shared_ptr<ucxx::Request> DataPlaneResources2::am_send_async(
    std::shared_ptr<ucxx::Endpoint> endpoint,
    const void* addr,
    std::size_t bytes,
    ucs_memory_type_t mem_type,
    std::optional<ucxx::AmReceiverCallbackInfo> callback_info,
    ucxx::RequestCallbackUserFunction callback_function,
    ucxx::RequestCallbackUserData callback_data)
{
    // TODO(MDD): Check that this EP belongs to this resource

    // Const cast away because UCXX only accepts void*
    auto request = endpoint->amSend(const_cast<void*>(addr),
                                    bytes,
                                    mem_type,
                                    callback_info,
                                    false,
                                    std::move(callback_function),
                                    std::move(callback_data));

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
    auto object_id = m_next_object_id++;

    return object_id;

    // // Lock on usages of the descriptor map
    // std::unique_lock lock(m_mutex);

    // while (m_remote_descriptor_by_id.find(object_id) != m_remote_descriptor_by_id.end())
    // {
    //     object_id = m_next_object_id++;
    // }

    // return object_id;
}

uint64_t DataPlaneResources2::register_remote_decriptor(
    std::shared_ptr<runtime::RemoteDescriptorImpl2> remote_descriptor)
{
    auto object_id = get_next_object_id();
    remote_descriptor->encoded_object().set_object_id(object_id);

    // Lock on usages of the descriptor map
    std::unique_lock lock(m_mutex);

    m_remote_descriptor_by_id[object_id] = remote_descriptor;

    // VLOG(10) << "DataPlaneResources2[" << this->get_instance_id() << "]: Registering RD(" << object_id
    //          << ") Tokens=" << remote_descriptor->encoded_object().tokens() << ".";

    return object_id;
}

channel::Egress<std::unique_ptr<runtime::RemoteDescriptor2>>& DataPlaneResources2::get_inbound_channel() const
{
    return *m_inbound_channel;
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

std::shared_ptr<runtime::RemoteDescriptorImpl2> DataPlaneResources2::get_descriptor(uint64_t object_id)
{
    return m_remote_descriptor_by_id.at(object_id);
}
}  // namespace mrc::data_plane

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

#include "internal/data_plane/client.hpp"

#include "internal/data_plane/callbacks.hpp"
#include "internal/data_plane/tags.hpp"
#include "internal/ucx/common.hpp"
#include "internal/ucx/context.hpp"
#include "internal/ucx/endpoint.hpp"
#include "internal/ucx/resources.hpp"
#include "internal/ucx/worker.hpp"
#include "internal/utils/contains.hpp"

#include "srf/channel/buffered_channel.hpp"
#include "srf/channel/channel.hpp"
#include "srf/channel/status.hpp"
#include "srf/codable/encode.hpp"
#include "srf/codable/encoded_object.hpp"
#include "srf/codable/protobuf_message.hpp"  // IWYU pragma: keep
#include "srf/exceptions/runtime_error.hpp"
#include "srf/memory/buffer_view.hpp"
#include "srf/memory/memory_kind.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/protos/codable.pb.h"
#include "srf/runnable/launch_control.hpp"
#include "srf/runnable/launcher.hpp"
#include "srf/runnable/runner.hpp"
#include "srf/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <boost/fiber/future/promise.hpp>
#include <glog/logging.h>
#include <ucp/api/ucp.h>
#include <ucp/api/ucp_def.h>
#include <ucs/memory/memory_type.h>
#include <ucs/type/status.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace srf::internal::data_plane {

Client::Client(resources::PartitionResourceBase& base, ucx::Resources& ucx) :
  resources::PartitionResourceBase(base),
  m_ucx(ucx)
{}

Client::~Client() = default;

void Client::register_instance(InstanceID instance_id, ucx::WorkerAddress worker_address)
{
    auto search = m_workers.find(instance_id);
    if (search != m_workers.end())
    {
        LOG(ERROR) << "instance_id: " << instance_id << " was already registered";
        throw std::runtime_error("instance_id already registered");
    }
    m_workers[instance_id] = std::move(worker_address);
}

const ucx::Endpoint& Client::endpoint(InstanceID id) const
{
    auto search_endpoints = m_endpoints.find(id);
    if (search_endpoints == m_endpoints.end())
    {
        auto search_workers = m_workers.find(id);
        if (search_workers == m_workers.end())
        {
            LOG(ERROR) << "no endpoint or worker addresss was found for instance_id: " << id;
            throw std::runtime_error("could not acquire ucx endpoint");
        }
        // lazy instantiation of the endpoint
        DVLOG(10) << "creating endpoint to instance_id: " << id;
        auto endpoint   = m_ucx.make_ep(search_workers->second);
        m_endpoints[id] = endpoint;
        return *endpoint;
    }
    DCHECK(search_endpoints->second);
    return *search_endpoints->second;
}

std::size_t Client::connections() const
{
    return m_endpoints.size();
}

void Client::async_recv(void* addr, std::size_t bytes, std::uint64_t tag, Request& request)
{
    static constexpr std::uint64_t mask = P2P_TAG & TAG_USER_MASK;  // NOLINT

    CHECK_EQ(request.m_request, nullptr);
    CHECK(request.m_state == Request::State::Init);
    request.m_state = Request::State::Running;

    ucp_request_param_t params;
    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.cb.recv      = Callbacks::recv;
    params.user_data    = &request;

    // build tag
    CHECK_LE(tag, TAG_USER_MASK);
    tag |= P2P_TAG;

    request.m_request = ucp_tag_recv_nbx(m_ucx.worker().handle(), addr, bytes, tag, mask, &params);
    CHECK(request.m_request);
    CHECK(!UCS_PTR_IS_ERR(request.m_request));
}

void Client::async_send(void* addr, std::size_t bytes, std::uint64_t tag, InstanceID instance_id, Request& request)
{
    CHECK_EQ(request.m_request, nullptr);
    CHECK(request.m_state == Request::State::Init);
    request.m_state = Request::State::Running;

    CHECK_LE(tag, TAG_USER_MASK);
    tag |= P2P_TAG;

    ucp_request_param_t send_params;
    send_params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    send_params.cb.send      = Callbacks::send;
    send_params.user_data    = &request;

    request.m_request = ucp_tag_send_nbx(endpoint(instance_id).handle(), addr, bytes, tag, &send_params);
    CHECK(request.m_request);
    CHECK(!UCS_PTR_IS_ERR(request.m_request));
}

void Client::async_get(void* addr,
                       std::size_t bytes,
                       InstanceID instance_id,
                       void* remote_addr,
                       const std::string& packed_remote_key,
                       Request& request)
{
    CHECK_EQ(request.m_request, nullptr);
    CHECK(request.m_state == Request::State::Init);
    request.m_state = Request::State::Running;

    const auto& ep = endpoint(instance_id);

    ucp_request_param_t params;
    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.cb.send      = Callbacks::send;
    params.user_data    = &request;

    {
        auto rc =
            ucp_ep_rkey_unpack(ep.handle(), packed_remote_key.data(), reinterpret_cast<ucp_rkey_h*>(&request.m_rkey));
        if (rc != UCS_OK)
        {
            LOG(ERROR) << "ucp_ep_rkey_unpack failed - " << ucs_status_string(rc);
            throw std::runtime_error("ucp_ep_rkey_unpack failed");
        }
    }

    request.m_request = ucp_get_nbx(ep.handle(),
                                    addr,
                                    bytes,
                                    reinterpret_cast<std::uint64_t>(remote_addr),
                                    reinterpret_cast<ucp_rkey_h>(request.m_rkey),
                                    &params);
    CHECK(request.m_request);
    CHECK(!UCS_PTR_IS_ERR(request.m_request));
}

// void Client::push_request(void* request)
// {
//     DCHECK(m_ucx_request_channel);
//     m_ucx_request_channel->await_write(std::move(request));
// }

// bool Client::is_connected_to(InstanceID instance_id) const
// {
//     return contains(m_workers, instance_id);
// }

// void Client::decrement_remote_descriptor(InstanceID id, ObjectID obj_id)
// {
//     ucp_tag_t tag = obj_id | DESCRIPTOR_TAG;
//     issue_network_event(id, tag);
// }

// void Client::issue_network_event(InstanceID id, ucp_tag_t tag)
// {
//     ucp_request_param_t params;
//     std::memset(&params, 0, sizeof(params));

//     auto* request = ucp_tag_send_nbx(endpoint(id).handle(), nullptr, 0, tag, &params);

//     if (request == nullptr /* UCS_OK */)
//     {
//         // send completed successfully
//         return;
//     }
//     if (UCS_PTR_IS_ERR(request))
//     {
//         LOG(ERROR) << "send failed";
//         throw std::runtime_error("send failed");
//     }

//     // send operation was scheduled by the ucx runtime
//     // adding requests to the channel will ensure the progress engine
//     // will work to make forward progress on queued network requests
//     push_request(std::move(request));
// }

// struct GetUserData
// {
//     Promise<void> promise;
//     ucp_rkey_h rkey;
// };

// static void rdma_get_callback(void* request, ucs_status_t status, void* user_data)
// {
//     DVLOG(1) << "rdma get callback start for request " << request;
//     auto* data = static_cast<GetUserData*>(user_data);
//     if (status != UCS_OK)
//     {
//         LOG(FATAL) << "rdma get failure occurred";
//         // data->promise.set_exception();
//     }
//     data->promise.set_value();
//     ucp_request_free(request);
//     ucp_rkey_destroy(data->rkey);
// }

/*
void Client::get(const protos::RemoteDescriptor& remote_md, Descriptor& buffer)
{
    CHECK_GE(buffer.size(), remote_md.remote_bytes());

    ucp_request_param_t params;
    std::memset(&params, 0, sizeof(params));

    auto* user_data  = new GetUserData;
    params.user_data = user_data;

    // unpack rkey on ep
    const auto& ep = endpoint(remote_md.instance_id());
    auto rc =
        ucp_ep_rkey_unpack(ep.handle(), reinterpret_cast<const void*>(remote_md.remote_key().data()), &user_data->rkey);
    if (rc != UCS_OK)
    {
        LOG(ERROR) << "ucp_ep_rkey_unpack failed - " << ucs_status_string(rc);
        throw exceptions::SrfRuntimeError("ucp_ep_rkey_unpack failed");
    }

    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.cb.send      = rdma_get_callback;

    // possibly set memory type if the fancy pointer provided those details
    switch (buffer.type())
    {
    case memory::memory_kind_type::host:
    case memory::memory_kind_type::pinned:
        params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMORY_TYPE;
        params.memory_type = UCS_MEMORY_TYPE_HOST;
        LOG(INFO) << "setting ucx_get_nbx memory type to HOST";
        break;
    case memory::memory_kind_type::device:
        params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMORY_TYPE;
        params.memory_type = UCS_MEMORY_TYPE_CUDA;
        LOG(FATAL) << "this path is probably broken";
        break;
    default:
        break;
    }

    auto future = user_data->promise.get_future();

    auto* status = ucp_get_nbx(
        ep.handle(), buffer.data(), remote_md.remote_bytes(), remote_md.remote_address(), user_data->rkey, &params);
    if (status == nullptr)  // UCS_OK
    {
        LOG(FATAL)
            << "should be unreachable";  // UCP_OP_ATTR_FLAG_NO_IMM_CMPL is set - should force the completion handler
    }
    else if (UCS_PTR_IS_ERR(status))
    {
        LOG(ERROR) << "rdma get failure";  // << ucs_status_string(status);
        throw exceptions::SrfRuntimeError("rdma get failure");
    }

    // await on promise
    future.get();
}
*/

// void Client::await_send(const InstanceID& instance_id,
//                         const PortAddress& port_address,
//                         const codable::EncodedObject& encoded_object)
// {
//     Promise<void> promise;
//     auto future = promise.get_future();

//     ucp_tag_t tag = port_address | INGRESS_TAG;
//     ucp_request_param_t params;

//     params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
//     params.cb.send      = send_completion_handler_with_future;
//     params.user_data    = &promise;

//     // serialize the proto of the encoded object into it's own encoded object
//     // dogfooding at its best
//     codable::EncodedObject msg;
//     codable::encode(encoded_object.proto(), msg);

//     // sanity check
//     // 1) there should be only 1 descriptor, and
//     // 2) the size of the memory block should be the size of the protos requested
//     DCHECK_EQ(msg.descriptor_count(), 1);
//     auto block = msg.memory_block(0);
//     DCHECK_EQ(block.bytes(), encoded_object.proto().ByteSizeLong());

//     // all encoded_objects are serialized to host memory
//     // these are small packed remote descriptors, not the actual payload data
//     params.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMORY_TYPE;
//     params.memory_type = UCS_MEMORY_TYPE_HOST;

//     // issue send
//     ucs_status_ptr_t request =
//         ucp_tag_send_nbx(endpoint(instance_id).handle(), block.data(), block.bytes(), tag, &params);

//     if (request == nullptr /* UCS_OK */)
//     {
//         return;
//     }
//     if (UCS_PTR_IS_ERR(request))
//     {
//         LOG(ERROR) << "send failed - ";
//         throw std::runtime_error("send failed");
//     }

//     // if we didn't complete immediate or throw an error, then the message
//     // is in flight. push the request to the progress engine which will
//     // wake up a progress fiber to complete the send
//     push_request(std::move(request));

//     // the caller of this await_send method will block and yield the fiber here
//     // the caller is calling an "await" method so blocking and yielding is implied
//     future.get();
// }

}  // namespace srf::internal::data_plane

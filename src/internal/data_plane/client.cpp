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
#include "internal/data_plane/resources.hpp"
#include "internal/data_plane/tags.hpp"
#include "internal/remote_descriptor/manager.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/ucx/endpoint.hpp"
#include "internal/ucx/resources.hpp"
#include "internal/ucx/worker.hpp"

#include "mrc/channel/buffered_channel.hpp"
#include "mrc/channel/channel.hpp"
#include "mrc/codable/protobuf_message.hpp"  // IWYU pragma: keep
#include "mrc/memory/literals.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/protos/codable.pb.h"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runnable/runner.hpp"
#include "mrc/runtime/remote_descriptor_handle.hpp"
#include "mrc/types.hpp"

#include <glog/logging.h>
#include <rxcpp/rx.hpp>
#include <ucp/api/ucp.h>
#include <ucs/type/status.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mrc::internal::data_plane {

using namespace mrc::memory::literals;

Client::Client(resources::PartitionResourceBase& base,
               ucx::Resources& ucx,
               control_plane::client::ConnectionsManager& connections_manager,
               memory::TransientPool& transient_pool) :
  resources::PartitionResourceBase(base),
  m_ucx(ucx),
  m_connnection_manager(connections_manager),
  m_transient_pool(transient_pool),
  m_rd_channel(std::make_unique<node::SourceChannelWriteable<RemoteDescriptorMessage>>())
{}

Client::~Client() = default;

std::shared_ptr<ucx::Endpoint> Client::endpoint_shared(const InstanceID& id) const
{
    auto search_endpoints = m_endpoints.find(id);
    if (search_endpoints == m_endpoints.end())
    {
        const auto& workers = m_connnection_manager.worker_addresses();
        auto search_workers = workers.find(id);
        if (search_workers == workers.end())
        {
            LOG(ERROR) << "no endpoint or worker addresss was found for instance_id: " << id;
            throw std::runtime_error("could not acquire ucx endpoint");
        }
        // lazy instantiation of the endpoint
        DVLOG(10) << "creating endpoint to instance_id: " << id;
        auto endpoint   = m_ucx.make_ep(search_workers->second);
        m_endpoints[id] = endpoint;
        return endpoint;
    }
    DCHECK(search_endpoints->second);
    return search_endpoints->second;
}

const ucx::Endpoint& Client::endpoint(const InstanceID& instance_id) const
{
    return *endpoint_shared(instance_id);
}

void Client::drop_endpoint(const InstanceID& instance_id)
{
    m_endpoints.erase(instance_id);
}

std::size_t Client::endpoint_count() const
{
    return m_endpoints.size();
}

void Client::async_recv(
    void* addr, std::size_t bytes, std::uint64_t tag, std::uint64_t mask, const ucx::Worker& worker, Request& request)
{
    CHECK_EQ(request.m_request, nullptr);
    CHECK(request.m_state == Request::State::Init);
    request.m_state = Request::State::Running;

    ucp_request_param_t params;
    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.cb.recv      = Callbacks::recv;
    params.user_data    = &request;

    request.m_request = ucp_tag_recv_nbx(worker.handle(), addr, bytes, tag, mask, &params);
    CHECK(request.m_request);
    CHECK(!UCS_PTR_IS_ERR(request.m_request));
}

void Client::async_p2p_recv(void* addr, std::size_t bytes, std::uint64_t tag, Request& request)
{
    static constexpr std::uint64_t mask = TAG_P2P_MSG & TAG_USER_MASK;  // NOLINT

    // build tag
    CHECK_LE(tag, TAG_USER_MASK);
    tag |= TAG_P2P_MSG;

    async_recv(addr, bytes, tag, mask, m_ucx.worker(), request);
}

void Client::async_send(
    void* addr, std::size_t bytes, std::uint64_t tag, const ucx::Endpoint& endpoint, Request& request)
{
    CHECK_EQ(request.m_request, nullptr);
    CHECK(request.m_state == Request::State::Init);
    request.m_state = Request::State::Running;

    ucp_request_param_t send_params;
    send_params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    send_params.cb.send      = Callbacks::send;
    send_params.user_data    = &request;

    request.m_request = ucp_tag_send_nbx(endpoint.handle(), addr, bytes, tag, &send_params);
    CHECK(request.m_request);
    CHECK(!UCS_PTR_IS_ERR(request.m_request));
}

void Client::async_p2p_send(
    void* addr, std::size_t bytes, std::uint64_t tag, InstanceID instance_id, Request& request) const
{
    CHECK_LE(tag, TAG_USER_MASK);
    tag |= TAG_P2P_MSG;

    async_send(addr, bytes, tag, endpoint(instance_id), request);
}

void Client::async_get(void* addr,
                       std::size_t bytes,
                       const ucx::Endpoint& ep,
                       std::uint64_t remote_addr,
                       ucp_rkey_h rkey,
                       Request& request)
{
    CHECK_EQ(request.m_request, nullptr);
    CHECK(request.m_state == Request::State::Init);
    request.m_state = Request::State::Running;

    ucp_request_param_t params;
    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.cb.send      = Callbacks::send;
    params.user_data    = &request;

    request.m_request = ucp_get_nbx(ep.handle(), addr, bytes, remote_addr, rkey, &params);
    CHECK(request.m_request);
    CHECK(!UCS_PTR_IS_ERR(request.m_request));
}

void Client::async_get(void* addr,
                       std::size_t bytes,
                       InstanceID instance_id,
                       void* remote_addr,
                       const std::string& packed_remote_key,
                       Request& request) const
{
    CHECK_EQ(request.m_request, nullptr);
    CHECK(request.m_state == Request::State::Init);

    const auto& ep = endpoint(instance_id);

    {
        auto rc =
            ucp_ep_rkey_unpack(ep.handle(), packed_remote_key.data(), reinterpret_cast<ucp_rkey_h*>(&request.m_rkey));
        if (rc != UCS_OK)
        {
            LOG(ERROR) << "ucp_ep_rkey_unpack failed - " << ucs_status_string(rc);
            throw std::runtime_error("ucp_ep_rkey_unpack failed");
        }
    }

    async_get(addr,
              bytes,
              ep,
              reinterpret_cast<std::uint64_t>(remote_addr),
              reinterpret_cast<ucp_rkey_h>(request.m_rkey),
              request);
}

void Client::async_am_send(
    std::uint32_t id, const void* header, std::size_t header_length, const ucx::Endpoint& endpoint, Request& request)
{
    CHECK_EQ(request.m_request, nullptr);
    CHECK(request.m_state == Request::State::Init);
    request.m_state = Request::State::Running;

    ucp_request_param_t params;
    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.cb.send      = Callbacks::send;
    params.user_data    = &request;

    request.m_request = ucp_am_send_nbx(endpoint.handle(), id, header, header_length, nullptr, 0, &params);
    CHECK(request.m_request);
    CHECK(!UCS_PTR_IS_ERR(request.m_request));
}

void Client::issue_remote_descriptor(RemoteDescriptorMessage&& msg)
{
    DCHECK(msg.rd);
    DCHECK(msg.endpoint);
    DCHECK_GT(msg.tag, 0);
    DCHECK_LE(msg.tag, TAG_USER_MASK);

    // detach handle from remote descriptor to ensure that the tokens are not decremented
    auto handle = remote_descriptor::Manager::unwrap_handle(std::move(msg.rd));

    // gain access to the protobuf backing the handle
    const auto& proto = handle->proto();

    auto msg_length = proto.ByteSizeLong();

    // todo(ryan) - parameterize mrc::data_plane::client::max_remote_descriptor_eager_size
    if (msg_length <= 1_MiB)
    {
        auto buffer = m_transient_pool.await_buffer(msg_length);
        CHECK(proto.SerializeToArray(buffer.data(), buffer.bytes()));

        // the message fits into the size of the preposted recvs issued by the data plane
        msg.tag |= TAG_EGR_MSG;

        Request request;
        async_send(buffer.data(), buffer.bytes(), msg.tag, *msg.endpoint, request);

        // await and yield the userspace thread until completed
        CHECK(request.await_complete());
    }
    else
    {
        LOG(FATAL) << "implement the rendez-vous path (nb_probe) on the server side";
    }
}

node::SourceChannelWriteable<RemoteDescriptorMessage>& Client::remote_descriptor_channel()
{
    CHECK(m_rd_channel);
    return *m_rd_channel;
}

void Client::do_service_start()
{
    CHECK(m_rd_channel);

    auto rd_writer = std::make_unique<node::RxSink<RemoteDescriptorMessage>>(
        [this](RemoteDescriptorMessage msg) { issue_remote_descriptor(std::move(msg)); });

    // todo(ryan) - parameterize mrc::data_plane::client::max_queued_remote_descriptor_sends
    rd_writer->update_channel(std::make_unique<channel::BufferedChannel<RemoteDescriptorMessage>>(128));

    // form edge
    mrc::node::make_edge(*m_rd_channel, *rd_writer);

    // todo(ryan) - parameterize mrc::data_plane::client::max_inflight_remote_descriptor_sends
    auto launch_options = Resources::launch_options(16);

    // launch rd_writer
    m_rd_writer = runnable().launch_control().prepare_launcher(launch_options, std::move(rd_writer))->ignition();
}

void Client::do_service_await_live()
{
    m_rd_writer->await_live();
}

void Client::do_service_stop()
{
    m_rd_channel.reset();
}

void Client::do_service_kill()
{
    m_rd_channel.reset();
    m_rd_writer->kill();
}

void Client::do_service_await_join()
{
    m_rd_writer->await_join();
}

}  // namespace mrc::internal::data_plane

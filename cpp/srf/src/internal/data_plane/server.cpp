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

#include "internal/data_plane/server.hpp"

#include "internal/data_plane/tags.hpp"

#include <srf/channel/status.hpp>
#include <srf/memory/block.hpp>
#include <srf/memory/memory_kind.hpp>
#include <srf/node/edge_builder.hpp>
#include <srf/node/operators/router.hpp>
#include <srf/node/source_channel.hpp>
#include <srf/runnable/context.hpp>
#include <srf/runnable/launch_control.hpp>
#include <srf/runnable/launch_options.hpp>
#include <srf/runnable/launcher.hpp>
#include <srf/runnable/runner.hpp>
#include <srf/types.hpp>
#include "internal/ucx/common.hpp"
#include "internal/ucx/context.hpp"
#include "internal/ucx/worker.hpp"

#include <glog/logging.h>
#include <ucp/api/ucp.h>
#include <ucs/type/status.h>
#include <boost/fiber/operations.hpp>
#include <rxcpp/rx-observer.hpp>
#include <rxcpp/rx-predef.hpp>
#include <rxcpp/rx-subscriber.hpp>
#include <rxcpp/rx.hpp>  // IWYU pragma: keep

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <ostream>
#include <utility>

namespace srf::internal::data_plane {

static thread_local rxcpp::subscriber<network_event_t>* static_subscriber = nullptr;

namespace {

void zero_bytes_completion_handler(void* request,
                                   ucs_status_t status,
                                   const ucp_tag_recv_info_t* msg_info,
                                   void* user_data)
{
    if (status != UCS_OK)
    {
        LOG(FATAL) << "zero_bytes_completion_handler observed " << ucs_status_string(status);
    }
    ucp_request_free(request);
}

void recv_completion_handler(void* request, ucs_status_t status, const ucp_tag_recv_info_t* msg_info, void* user_data)
{
    if (status != UCS_OK)
    {
        LOG(FATAL) << "recv_completion_handler observed " << ucs_status_string(status);
    }
    auto port_address = tag_decode_user_tag(msg_info->sender_tag);
    DCHECK(static_subscriber && static_subscriber->is_subscribed());
    auto msg = std::make_pair(port_address, memory::block(user_data, msg_info->length, memory::memory_kind_type::host));
    static_subscriber->on_next(std::move(msg));
    ucp_request_free(request);
}

}  // namespace

Server::Server(std::shared_ptr<ucx::Context> context, std::shared_ptr<resources::PartitionResources> resources) :
  m_worker(std::make_shared<ucx::Worker>(context))
{}

Server::~Server()
{
    Service::call_in_destructor();
}

void Server::do_service_start()
{
    m_deserialize_source = std::make_shared<node::Router<PortAddress, memory::block>>();
    m_rd_source          = std::make_unique<node::SourceChannelWriteable<ucp_tag_t>>();

    auto progress_engine = std::make_unique<DataPlaneServerWorker>(m_worker);
    node::make_edge(*progress_engine, *m_deserialize_source);

    // all network runnables use the `srf_network` engine factory
    DVLOG(10) << "launch network event mananger progress engine";
    LOG(FATAL) << "get launch control from partition resources";
    //  m_progress_engine =
    //        launch_control.prepare_launcher(runnable::LaunchOptions("srf_network"),
    //        std::move(progress_engine))->ignition();
}

void Server::do_service_await_live()
{
    m_progress_engine->await_live();
}

void Server::do_service_stop()
{
    m_progress_engine->stop();
}

void Server::do_service_kill()
{
    m_progress_engine->kill();
}

void Server::do_service_await_join()
{
    m_progress_engine->await_join();
    if (m_rd_source)
    {
        m_rd_source.reset();
    }
}

ucx::WorkerAddress Server::worker_address() const
{
    return m_worker->address();
}

node::Router<PortAddress, memory::block>& Server::deserialize_source()
{
    CHECK(m_deserialize_source);
    return *m_deserialize_source;
}

// NetworkEventProgressEngine

DataPlaneServerWorker::DataPlaneServerWorker(Handle<ucx::Worker> worker) : m_worker(std::move(worker)) {}

void DataPlaneServerWorker::data_source(rxcpp::subscriber<network_event_t>& s)
{
    ucp_tag_message_h msg;
    ucp_tag_recv_info_t msg_info;
    std::uint32_t backoff = 1;

    // set static variable for callbacks
    static_subscriber = &s;

    while (true)
    {
        for (;;)
        {
            msg = ucp_tag_probe_nb(m_worker->handle(), m_tag, m_tag_mask, 1, &msg_info);
            if (!s.is_subscribed())
            {
                return;
            }
            if (msg != nullptr)
            {
                break;
            }
            while (m_worker->progress() != 0U)
            {
                backoff = 1;
            }

            boost::this_fiber::yield();

            /*
            if (backoff < 1048576)
            {
                backoff = backoff << 1;
            }
            if (backoff < 32768)
            {
                boost::this_fiber::yield();
            }
            else
            {
                boost::this_fiber::sleep_for(std::chrono::nanoseconds(backoff));
            }
            */
        }

        on_tagged_msg(s, msg, msg_info);
        backoff = 1;
    }
}

void DataPlaneServerWorker::on_tagged_msg(rxcpp::subscriber<network_event_t>& subscriber,
                                          ucp_tag_message_h msg,
                                          const ucp_tag_recv_info_t& msg_info)
{
    ucp_request_param_t params;
    // std::memset(&params, 0, sizeof(params));
    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
    params.cb.recv      = zero_bytes_completion_handler;
    params.user_data    = nullptr;

    // receive buffer
    void* recv_addr        = nullptr;
    std::size_t recv_bytes = 0;

    auto msg_type = tag_decode_msg_type(msg_info.sender_tag);

    switch (msg_type)
    {
    case INGRESS_TAG: {
        params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |   // recv_completion_handler
                              UCP_OP_ATTR_FIELD_USER_DATA |  // user_data
                              UCP_OP_ATTR_FIELD_RECV_INFO |  // not sure if this is needed
                              UCP_OP_ATTR_FLAG_NO_IMM_CMPL;  // force the completion handler to be used

        // todo(#139) - replace malloc with an unexpected receive / transactional memory_resource for
        // ultra fast allocation
        // in that work create a struct InFlightEncodedObject and allocate msg_info.length +
        // sizeof(InFightEncodedObject) + alignment, and pass the starting pointer of that block to the user_data void*
        // of the callback. Then unpack the struct and blob in the callback. The pointer to the subscriber is part of
        // the InFlightEncodedObject which will remove the thread local storage pointer we are doing now.

        recv_bytes       = msg_info.length;
        recv_addr        = std::malloc(recv_bytes);
        params.user_data = recv_addr;
        params.cb.recv   = recv_completion_handler;
        break;
    }
    case DESCRIPTOR_TAG:
        // m_rd_source.await_write(msg_info.sender_tag);
        // m_descriptors_channel->await_write(msg_info.sender_tag);
        LOG(FATAL) << "remote descriptor not implemented";
        break;

    case FUTURE_TAG:
        LOG(FATAL) << "remote futures/promises not implemented";
        break;

    default:
        LOG(FATAL) << "unknown network event received: " << msg_info.sender_tag;
    };

    void* status = ucp_tag_msg_recv_nbx(m_worker->handle(), recv_addr, recv_bytes, msg, &params);
    if (UCS_PTR_IS_ERR(status))
    {
        LOG(FATAL) << "ucp_tag_msg_recv_nbx for 0-byte event failed";
    }
}

}  // namespace srf::internal::data_plane

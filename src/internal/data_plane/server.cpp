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
#include "internal/runnable/resources.hpp"
#include "internal/ucx/common.hpp"
#include "internal/ucx/resources.hpp"
#include "internal/ucx/worker.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/memory/literals.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/generic_source.hpp"
#include "mrc/node/operators/router.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runnable/runner.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <boost/fiber/operations.hpp>
#include <glog/logging.h>
#include <rxcpp/rx.hpp>
#include <ucp/api/ucp.h>
#include <ucp/api/ucp_def.h>
#include <ucs/type/status.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <utility>

namespace mrc::internal::data_plane {

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

static void pre_post_recv(detail::PrePostedRecvInfo* info);

void pre_posted_recv_callback(void* request, ucs_status_t status, const ucp_tag_recv_info_t* msg_info, void* user_data)
{
    DCHECK(user_data);
    auto* info = static_cast<detail::PrePostedRecvInfo*>(user_data);
    if (status == UCS_OK)  // cpp20 [[likely]]
    {
        // grab tag and free request - not sure if there will be a race condition on msg_info
        auto tag    = decode_user_bits(msg_info->sender_tag);
        auto length = msg_info->length;
        ucp_request_free(request);

        // create a shallow copy of the transient buffer with received buffer size
        DCHECK_LE(length, info->buffer.bytes());
        memory::TransientBuffer buffer(info->buffer.data(), length, info->buffer);

        // write tag to channel - create a shallow copy of the transient buffer with received buffer size
        info->channel->await_write(std::make_pair(tag, std::move(buffer)));

        // create a new transient buffer
        info->buffer = info->pool->await_buffer(info->buffer.bytes());

        // repost recv
        pre_post_recv(info);
    }
    else if (status == UCS_ERR_CANCELED)
    {
        ucp_request_free(info->request);
        info->request = nullptr;  // this ensures than cancel will not be called again if a kill is issued after stop
    }
    else
    {
        LOG(FATAL) << "data_plane: pre_posted_recv_callback failed with status: " << ucs_status_string(status);
    }
}

void pre_post_recv(detail::PrePostedRecvInfo* info)
{
    ucp_request_param_t params;
    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.cb.recv      = pre_posted_recv_callback;
    params.user_data    = info;

    // match upper 16 bits instead of only upper 4 bits
    info->request =
        ucp_tag_recv_nbx(info->worker, info->buffer.data(), info->buffer.bytes(), TAG_EGR_MSG, TAG_MSG_MASK, &params);
    CHECK(info->request);
    CHECK(!UCS_PTR_IS_ERR(info->request));
}

}  // namespace

using namespace mrc::memory::literals;

class DataPlaneServerWorker final : public node::GenericSource<network_event_t>
{
  public:
    DataPlaneServerWorker(ucx::Worker& worker);

  private:
    void data_source(rxcpp::subscriber<network_event_t>& s) final;

    void on_tagged_msg(rxcpp::subscriber<network_event_t>& subscriber,
                       ucp_tag_message_h msg,
                       const ucp_tag_recv_info_t& msg_info);

    ucx::Worker& m_worker;

    // modify these to adjust the tag matching
    // 0/0 is the equivalent of match all tags
    ucp_tag_t m_tag{0};
    ucp_tag_t m_tag_mask{0};
};

Server::Server(resources::PartitionResourceBase& provider,
               ucx::Resources& ucx,
               memory::HostResources& host,
               memory::TransientPool& transient_pool,
               InstanceID instance_id) :
  resources::PartitionResourceBase(provider),
  m_ucx(ucx),
  m_host(host),
  m_instance_id(instance_id),
  m_transient_pool(transient_pool)
{}

Server::~Server()
{
    Service::call_in_destructor();
}

void Server::do_service_start()
{
    m_ucx.network_task_queue()
        .enqueue([this] {
            // source channel ucx tag recvs masked with the RemoteDescriptor tag
            // this recv has no recv payload, we simply write the tag to the channel
            m_prepost_channel = std::make_unique<node::SourceChannelWriteable<network_event_t>>();

            m_pre_posted_recv_info.resize(m_pre_posted_recv_count);
            for (auto& info : m_pre_posted_recv_info)
            {
                info.worker  = m_ucx.worker().handle();
                info.channel = m_prepost_channel.get();
                info.pool    = &m_transient_pool;
                info.buffer  = m_transient_pool.await_buffer(1_MiB);
                pre_post_recv(&info);
            }

            // source for ucx tag recvs with data
            auto progress_engine = std::make_unique<DataPlaneServerWorker>(m_ucx.worker());

            // router for ucx tag recvs with data
            m_deserialize_source = std::make_shared<node::Router<PortAddress, memory::TransientBuffer>>();
            node::make_edge(*m_prepost_channel, *m_deserialize_source);

            // for edge between source and router - on channel operator driven by the source thread
            node::make_edge(*progress_engine, *m_deserialize_source);

            // all network runnables use the `mrc_network` engine factory
            DVLOG(10) << "launch network event mananger progress engine";
            m_progress_engine =
                runnable()
                    .launch_control()
                    .prepare_launcher(mrc::runnable::LaunchOptions("mrc_network"), std::move(progress_engine))
                    ->ignition();
        })
        .get();
}

void Server::do_service_await_live()
{
    m_progress_engine->await_live();
}

void Server::do_service_stop()
{
    DVLOG(10) << "data_plane server: stop issued";

    m_ucx.network_task_queue()
        .enqueue([this] {
            // we need to cancel all preposted recvs before shutting down the progress engine
            DVLOG(10) << "data_plane server: cancelling all outstanding pre-posted recvs";
            for (auto& info : m_pre_posted_recv_info)
            {
                if (info.request != nullptr)
                {
                    ucp_request_cancel(m_ucx.worker().handle(), info.request);
                }

                // we are on the network task queue thread, so we can pump the progress engine until
                // the cancelled request is complete
                while (info.request != nullptr)
                {
                    m_ucx.worker().progress();
                }
            }
        })
        .get();

    DVLOG(10) << "data_plane server: issuing stop to progress engine runnable";
    m_progress_engine->stop();
}

void Server::do_service_kill()
{
    m_progress_engine->kill();
}

void Server::do_service_await_join()
{
    m_progress_engine->await_join();
    if (m_prepost_channel)
    {
        m_prepost_channel.reset();
    }
}

ucx::WorkerAddress Server::worker_address() const
{
    return m_ucx.worker().address();
}

node::Router<std::uint64_t, memory::TransientBuffer>& Server::deserialize_source()
{
    CHECK(m_deserialize_source);
    return *m_deserialize_source;
}

// NetworkEventProgressEngine

DataPlaneServerWorker::DataPlaneServerWorker(ucx::Worker& worker) : m_worker(worker) {}

void DataPlaneServerWorker::data_source(rxcpp::subscriber<network_event_t>& s)
{
    ucp_tag_message_h msg;
    ucp_tag_recv_info_t msg_info;
    std::uint32_t backoff = 1;

    DVLOG(10) << "starting data plane server progress engine loop";

    // the progress loop has tag_probe_nb disabled
    // this should be re-enabled to accept tagged messages that have payloads
    // larger than the pre-posted recv buffers

    while (true)
    {
        for (;;)
        {
            // msg = ucp_tag_probe_nb(m_worker->handle(), m_tag, m_tag_mask, 1, &msg_info);
            if (!s.is_subscribed())
            {
                DVLOG(10) << "exiting data plane server progress engine loop";
                return;
            }
            // if (msg != nullptr)
            // {
            //     break;
            // }
            while (m_worker.progress() != 0U)
            {
                backoff = 1;
            }

            boost::this_fiber::yield();
        }

        // on_tagged_msg(s, msg, msg_info);
        // backoff = 1;
    }
}

void DataPlaneServerWorker::on_tagged_msg(rxcpp::subscriber<network_event_t>& subscriber,
                                          ucp_tag_message_h msg,
                                          const ucp_tag_recv_info_t& msg_info)
{
    ucp_request_param_t params;
    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
    params.cb.recv      = zero_bytes_completion_handler;
    params.user_data    = nullptr;

    // receive buffer
    void* recv_addr        = nullptr;
    std::size_t recv_bytes = 0;

    auto msg_type = decode_tag_msg(msg_info.sender_tag);

    switch (msg_type)
    {
    case TAG_RND_MSG: {
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
        // params.cb.recv   = recv_completion_handler;
        break;
    }

    default:
        LOG(FATAL) << "unknown network event received: " << msg_info.sender_tag;
    };

    void* status = ucp_tag_msg_recv_nbx(m_worker.handle(), recv_addr, recv_bytes, msg, &params);
    if (UCS_PTR_IS_ERR(status))
    {
        LOG(FATAL) << "ucp_tag_msg_recv_nbx for 0-byte event failed";
    }
}

}  // namespace mrc::internal::data_plane

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

#include "internal/ucx/all.hpp"
#include "internal/ucx/endpoint.hpp"

#include "mrc/channel/forward.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <boost/fiber/future/future_status.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <ucp/api/ucp.h>
#include <ucs/type/status.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <list>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>

using namespace mrc;
using namespace internal::ucx;

class TestUCX : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_context = std::make_shared<Context>();
    }
    void TearDown() override {}

  public:
    Handle<Context> m_context;
};

TEST_F(TestUCX, Init) {}

TEST_F(TestUCX, CreateWorker)
{
    auto worker = std::make_shared<Worker>(m_context);
    worker->progress();
}

TEST_F(TestUCX, CreateWorkerAddress)
{
    auto worker  = std::make_shared<Worker>(m_context);
    auto address = worker->address();
    EXPECT_GT(address.length(), 0);
}

TEST_F(TestUCX, EndpointsInProcess)
{
    auto worker_1 = std::make_shared<Worker>(m_context);
    auto worker_2 = std::make_shared<Worker>(m_context);

    // get worker 1's address
    auto worker_1_addr = worker_1->address();

    // worker 2 will create an endpoint to worker 1
    auto ep_w2 = worker_2->create_endpoint(worker_1_addr);

    worker_1->progress();
    worker_2->progress();
}

struct GetUserData
{
    Promise<void> promise;
    ucp_rkey_h rkey;
};

static void rdma_get_callback(void* request, ucs_status_t status, void* user_data)
{
    DVLOG(1) << "rdma get callback start for request " << request;
    auto* data = static_cast<GetUserData*>(user_data);
    if (status != UCS_OK)
    {
        LOG(FATAL) << "rdma get failure occurred";
        // data->promise.set_exception();
    }
    data->promise.set_value();
    ucp_request_free(request);
    ucp_rkey_destroy(data->rkey);
}

TEST_F(TestUCX, Get)
{
    auto context = std::make_shared<Context>();

    auto worker_get_src = std::make_shared<Worker>(context);
    auto worker_get_dst = std::make_shared<Worker>(context);

    // create an ep from the worker issuing the GET to the worker address that is the source bytes
    auto ep = std::make_shared<Endpoint>(worker_get_dst, worker_get_src->address());

    auto* src_data                             = std::malloc(4096);
    auto [src_lkey, src_rbuff, src_rbuff_size] = context->register_memory_with_rkey(src_data, 4096);

    auto* dst_data = std::malloc(4096);
    auto* dst_lkey = context->register_memory(dst_data, 4096);

    auto* user_data = new GetUserData;
    auto rc         = ucp_ep_rkey_unpack(ep->handle(), src_rbuff, &user_data->rkey);
    if (rc != UCS_OK)
    {
        LOG(ERROR) << "ucp_ep_rkey_unpack failed - " << ucs_status_string(rc);
        throw std::runtime_error("ucp_ep_rkey_unpack failed");
    }

    ucp_request_param_t params;
    std::memset(&params, 0, sizeof(params));

    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.cb.send      = rdma_get_callback;
    params.user_data    = user_data;

    auto future = user_data->promise.get_future();

    auto* get_rc =
        ucp_get_nbx(ep->handle(), dst_data, 4096, reinterpret_cast<std::uint64_t>(src_data), user_data->rkey, &params);
    if (get_rc == nullptr /* UCS_OK */)
    {
        LOG(FATAL)
            << "should be unreachable";  // UCP_OP_ATTR_FLAG_NO_IMM_CMPL is set - should force the completion handler
    }
    else if (UCS_PTR_IS_ERR(get_rc))
    {
        LOG(ERROR) << "rdma get failure";  // << ucs_status_string(status);
        throw std::runtime_error("rdma get failure");
    }

    auto status = boost::fibers::future_status::timeout;
    while (status == boost::fibers::future_status::timeout)
    {
        status = future.wait_for(std::chrono::milliseconds(1));
        worker_get_dst->progress();
        worker_get_src->progress();
    }
    future.get();

    // unregister memory
    ucp_rkey_buffer_release(src_rbuff);
    context->unregister_memory(src_lkey);
    context->unregister_memory(dst_lkey);
}

// Recv

static void recv_get_callback(void* request, ucs_status_t status, const ucp_tag_recv_info_t* msg_info, void* user_data)
{
    if (status != UCS_OK)
    {
        LOG(FATAL) << "zero_bytes_completion_handler observed " << ucs_status_string(status);
    }
    if (user_data != nullptr)
    {
        auto* promise = static_cast<std::promise<ucp_tag_t>*>(user_data);
        promise->set_value(msg_info->sender_tag);
    }
    ucp_request_free(request);
}

static void send_callback(void* request, ucs_status_t status, void* user_data)
{
    DVLOG(1) << "send callback start for request " << request;
    if (status != UCS_OK)
    {
        LOG(FATAL) << "rdma get failure occurred";
        // data->promise.set_exception();
    }
    auto* promise = static_cast<std::promise<void>*>(user_data);
    promise->set_value();
    ucp_request_free(request);
}

class SendRecvManager
{
  private:
    bool m_is_subscribed{true};
    std::list<void*> m_send_requests;
    std::list<void*> m_recv_requests;
};

TEST_F(TestUCX, Recv)
{
    auto context = std::make_shared<Context>();

    auto worker_src = std::make_shared<Worker>(context);
    auto worker_dst = std::make_shared<Worker>(context);

    // create an ep from the worker issuing the GET to the worker address that is the source bytes
    auto ep = std::make_shared<Endpoint>(worker_src, worker_dst->address());

    std::promise<void> send_promise;
    std::promise<ucp_tag_t> recv_promise;
    auto send_future = send_promise.get_future();
    auto recv_future = recv_promise.get_future();

    // dst issue recv
    ucp_request_param_t params;
    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.cb.recv      = recv_get_callback;
    params.user_data    = &recv_promise;

    void* status_ptr_1 = ucp_tag_recv_nbx(worker_dst->handle(), nullptr, 0, 0, 0, &params);
    EXPECT_NE(status_ptr_1, nullptr);

    // void* status_ptr_2 = ucp_tag_recv_nbx(worker_dst->handle(), nullptr, 0, 0, 0, &params);
    // EXPECT_NE(status_ptr_2, nullptr);

    // sender
    ucp_request_param_t send_params;
    send_params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    send_params.cb.send      = send_callback;
    send_params.user_data    = &send_promise;

    ucs_status_ptr_t send_rc = ucp_tag_send_nbx(ep->handle(), nullptr, 0, 42, &send_params);
    EXPECT_NE(send_rc, nullptr);

    auto send_status = std::future_status::timeout;
    while (send_status == std::future_status::timeout)
    {
        send_status = send_future.wait_for(std::chrono::microseconds(1));
        worker_dst->progress();
        worker_src->progress();
    }
    send_future.get();

    VLOG(1) << "send complete";

    auto recv_status = std::future_status::timeout;
    while (recv_status == std::future_status::timeout)
    {
        recv_status = recv_future.wait_for(std::chrono::microseconds(1));
        worker_dst->progress();
        worker_src->progress();
    }

    auto tag = recv_future.get();

    EXPECT_EQ(tag, 42);

    // note: uncommenting this code will push a recv request but the worker and ep will shutdown
    // resulting in a ucx working message
    // if we have pre-posted recvs, we will need a way to cancel them to prevent these warnings
    // {
    //     ucp_request_param_t params;
    //     params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA |
    //     UCP_OP_ATTR_FLAG_NO_IMM_CMPL; params.cb.recv      = recv_get_callback; params.user_data    = nullptr;

    //     void* status_ptr_1 = ucp_tag_recv_nbx(worker_dst->handle(), nullptr, 0, 0, 0, &params);
    //     EXPECT_NE(status_ptr_1, nullptr);
    // }
}

// Recv

TEST_F(TestUCX, Recv2)
{
    auto context = std::make_shared<Context>();

    auto worker_src = std::make_shared<Worker>(context);
    auto worker_dst = std::make_shared<Worker>(context);

    // create an ep from the worker issuing the GET to the worker address that is the source bytes
    auto ep = std::make_shared<Endpoint>(worker_src, worker_dst->address());

    std::promise<void> send_promise;
    std::promise<ucp_tag_t> recv_promise;
    auto send_future = send_promise.get_future();
    auto recv_future = recv_promise.get_future();

    // dst issue recv
    ucp_request_param_t params;
    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.cb.recv      = recv_get_callback;
    params.user_data    = &recv_promise;

    void* status_ptr_1 = ucp_tag_recv_nbx(worker_dst->handle(), nullptr, 0, 0, 0, &params);
    EXPECT_NE(status_ptr_1, nullptr);

    // void* status_ptr_2 = ucp_tag_recv_nbx(worker_dst->handle(), nullptr, 0, 0, 0, &params);
    // EXPECT_NE(status_ptr_2, nullptr);

    // sender
    ucp_request_param_t send_params;
    send_params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    send_params.cb.send      = send_callback;
    send_params.user_data    = &send_promise;

    ucs_status_ptr_t send_rc = ucp_tag_send_nbx(ep->handle(), nullptr, 0, 42, &send_params);
    EXPECT_NE(send_rc, nullptr);

    auto send_status = std::future_status::timeout;
    while (send_status == std::future_status::timeout)
    {
        send_status = send_future.wait_for(std::chrono::microseconds(1));
        worker_dst->progress();
        worker_src->progress();
    }
    send_future.get();

    VLOG(1) << "send complete";

    auto recv_status = std::future_status::timeout;
    while (recv_status == std::future_status::timeout)
    {
        recv_status = recv_future.wait_for(std::chrono::microseconds(1));
        worker_dst->progress();
        worker_src->progress();
    }

    auto tag = recv_future.get();

    EXPECT_EQ(tag, 42);

    // note: uncommenting this code will push a recv request but the worker and ep will shutdown
    // resulting in a ucx working message
    // if we have pre-posted recvs, we will need a way to cancel them to prevent these warnings
    // {
    //     ucp_request_param_t params;
    //     params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA |
    //     UCP_OP_ATTR_FLAG_NO_IMM_CMPL; params.cb.recv      = recv_get_callback; params.user_data    = nullptr;

    //     void* status_ptr_1 = ucp_tag_recv_nbx(worker_dst->handle(), nullptr, 0, 0, 0, &params);
    //     EXPECT_NE(status_ptr_1, nullptr);
    // }
}

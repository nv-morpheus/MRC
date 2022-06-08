/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "testing.grpc.pb.h"
#include "testing.pb.h"

#include "nvrpc/client/client_streaming_v3.h"
#include "nvrpc/client/client_unary.h"
#include "nvrpc/client/executor.h"

#include <glog/logging.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/security/credentials.h>

#include <functional>
#include <future>
#include <memory>   // for shared_ptr, make_shared
#include <ostream>  // for logging
#include <utility>  // for move

namespace nvrpc {
namespace testing {

template <typename Request, typename Response>
using streaming_client_t = typename client::v3::ClientStreaming<Request, Response>;

template <typename Request, typename Response>
class ClientStreamingV3 : public streaming_client_t<Request, Response>
{
    using base_client_t = streaming_client_t<Request, Response>;

  public:
    using PrepareFn      = typename base_client_t::PrepareFn;
    using CallbackOnSend = std::function<void(Request&&)>;
    using CallbackOnRecv = std::function<void(Response&&)>;

    ClientStreamingV3(PrepareFn prepare_fn,
                      std::shared_ptr<client::Executor> executor,
                      CallbackOnSend send_callback,
                      CallbackOnRecv recv_callback) :
      base_client_t(prepare_fn, executor),
      m_send_callback(send_callback),
      m_recv_callback(recv_callback)
    {}
    virtual ~ClientStreamingV3() override {}

    void CallbackOnResponseReceived(Response&& response) final override
    {
        m_recv_callback(std::move(response));
    }

    void CallbackOnRequestSent(Request&& request) final override
    {
        m_send_callback(std::move(request));
    }

    void CallbackOnComplete(const ::grpc::Status& status) final override
    {
        LOG(INFO) << "Status: " << (status.ok() ? "OK" : "CANCELLED");
        m_promise.set_value(status);
    }

    std::shared_future<::grpc::Status> Done()
    {
        this->CloseWrites();
        return m_promise.get_future().share();
    }

  private:
    std::promise<::grpc::Status> m_promise;
    std::function<void(Request&&)> m_send_callback;
    std::function<void(Response&&)> m_recv_callback;
};

std::unique_ptr<client::ClientUnary<Input, Output>> BuildUnaryClient()
{
    auto executor = std::make_shared<client::Executor>(1);

    auto channel = grpc::CreateChannel("localhost:13377", grpc::InsecureChannelCredentials());
    std::shared_ptr<TestService::Stub> stub = TestService::NewStub(channel);

    auto infer_prepare_fn = [stub](
        ::grpc::ClientContext * context, const Input& request, ::grpc::CompletionQueue* cq) -> auto
    {
        return std::move(stub->PrepareAsyncUnary(context, request, cq));
    };

    return std::make_unique<client::ClientUnary<Input, Output>>(infer_prepare_fn, executor);
}

std::unique_ptr<ClientStreamingV3<Input, Output>> BuildStreamingClient(std::function<void(Input&&)> on_sent,
                                                                       std::function<void(Output&&)> on_recv)
{
    auto executor = std::make_shared<client::Executor>(1);

    auto channel = grpc::CreateChannel("localhost:13377", grpc::InsecureChannelCredentials());
    std::shared_ptr<TestService::Stub> stub = TestService::NewStub(channel);

    auto infer_prepare_fn = [stub](::grpc::ClientContext * context, ::grpc::CompletionQueue * cq) -> auto
    {
        return std::move(stub->PrepareAsyncStreaming(context, cq));
    };

    return std::make_unique<ClientStreamingV3<Input, Output>>(infer_prepare_fn, executor, on_sent, on_recv);
}

}  // namespace testing
}  // namespace nvrpc

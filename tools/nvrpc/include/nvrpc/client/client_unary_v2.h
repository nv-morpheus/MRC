/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <functional>
#include <future>
#include <memory>

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include "nvrpc/client/base_context.h"
#include "nvrpc/client/executor.h"
#include "srf/core/async_compute.h"

namespace nvrpc {
namespace client {
namespace v2 {
template <typename Request, typename Response>
struct ClientUnary : public BaseContext
{
    using Client = ClientUnary<Request, Response>;
    using Reader = std::unique_ptr<::grpc::ClientAsyncResponseReader<Response>>;

  public:
    using PrepareFn = std::function<Reader(::grpc::ClientContext*, const Request&, ::grpc::CompletionQueue*)>;

    ClientUnary(PrepareFn prepare_fn, std::shared_ptr<Executor> executor) :
      m_PrepareFn(prepare_fn),
      m_Executor(executor)
    {
        m_NextState = &Client::StateInvalid;
    }

    ~ClientUnary() {}

    void Write(Request&&);

    virtual void CallbackOnRequestSent(Request&&) {}
    virtual void CallbackOnResponseReceived(Response&&)    = 0;
    virtual void CallbackOnComplete(const ::grpc::Status&) = 0;

    bool ExecutorShouldDeleteContext() const override
    {
        return false;
    }

  protected:
    ::grpc::ClientContext& GetClientContext()
    {
        return m_Context;
    }

  private:
    PrepareFn m_PrepareFn;
    std::shared_ptr<Executor> m_Executor;

    ::grpc::Status m_Status;
    ::grpc::ClientContext m_Context;
    Reader m_Stream;

    Request m_Request;
    Response m_Response;

    bool RunNextState(bool ok) final override
    {
        return (this->*m_NextState)(ok);
    }

    bool StateFinishDone(bool);
    bool StateInvalid(bool);

    bool (Client::*m_NextState)(bool);
};

template <typename Request, typename Response>
void ClientUnary<Request, Response>::Write(Request&& request)
{
    CHECK(m_Stream == nullptr);

    m_Request   = std::move(request);
    m_NextState = &Client::StateFinishDone;

    m_Stream = m_PrepareFn(&m_Context, m_Request, m_Executor->GetNextCQ());
    m_Stream->StartCall();
    m_Stream->Finish(&m_Response, &m_Status, this->Tag());
}

template <typename Request, typename Response>
bool ClientUnary<Request, Response>::StateFinishDone(bool ok)
{
    m_NextState = &Client::StateInvalid;

    if (!ok)
    {
        DVLOG(1) << "FinishDone handler called with NOT OK";
    }

    DVLOG(1) << "calling on complete callback";
    if (m_Status.ok())
    {
        CallbackOnResponseReceived(std::move(m_Response));
    }
    CallbackOnComplete(m_Status);

    return false;
}

template <typename Request, typename Response>
bool ClientUnary<Request, Response>::StateInvalid(bool ok)
{
    LOG(FATAL) << "logic error in ClientUnary state management";
    return false;
}

}  // namespace v2
}  // namespace client
}  // namespace nvrpc

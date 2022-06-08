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

template <typename Request, typename Response>
struct ClientSingleUpMultipleDown : public BaseContext
{
    using Client = ClientSingleUpMultipleDown<Request, Response>;

  public:
    using PrepareFn = std::function<std::unique_ptr<::grpc::ClientAsyncReader<Response>>(
        ::grpc::ClientContext*, const Request&, ::grpc::CompletionQueue*)>;

    ClientSingleUpMultipleDown(PrepareFn prepare_fn, std::shared_ptr<Executor> executor) :
      m_PrepareFn(prepare_fn),
      m_Executor(executor)
    {
        m_NextState = &Client::StateInvalid;
    }

    ~ClientSingleUpMultipleDown() {}

    void Write(Request&&);
    void Cancel()
    {
        m_Context.TryCancel();
    }

    bool ExecutorShouldDeleteContext() const final override
    {
        return false;
    }

  protected:
    ::grpc::ClientContext& GetClientContext()
    {
        return m_Context;
    }

  private:
    virtual void CallbackOnRequestSent(Request&&) {}
    virtual void CallbackOnResponseReceived(Response&&)    = 0;
    virtual void CallbackOnComplete(const ::grpc::Status&) = 0;

    PrepareFn m_PrepareFn;
    std::shared_ptr<Executor> m_Executor;

    ::grpc::Status m_Status;
    ::grpc::ClientContext m_Context;
    std::unique_ptr<::grpc::ClientAsyncReader<Response>> m_Stream;

    Request m_Request;
    Response m_Response;

    bool RunNextState(bool ok) final override
    {
        return (this->*m_NextState)(ok);
    }

    bool StateStreamInitialized(bool);
    bool StateReadDone(bool);
    bool StateFinishDone(bool);
    bool StateInvalid(bool);

    bool (Client::*m_NextState)(bool);
};

template <typename Request, typename Response>
void ClientSingleUpMultipleDown<Request, Response>::Write(Request&& request)
{
    CHECK(m_Stream == nullptr);

    m_Request   = std::move(request);
    m_NextState = &Client::StateStreamInitialized;

    m_Stream = m_PrepareFn(&m_Context, m_Request, m_Executor->GetNextCQ());
    m_Stream->StartCall(this->Tag());
}

template <typename Request, typename Response>
bool ClientSingleUpMultipleDown<Request, Response>::StateStreamInitialized(bool ok)
{
    if (!ok)
    {
        DVLOG(1) << "stream failed to initialize";
        return false;
    }

    DVLOG(1) << "executing callback after initial write to server finished";
    CallbackOnRequestSent(std::move(m_Request));

    m_NextState = &Client::StateReadDone;
    m_Stream->Read(&m_Response, this->Tag());
}

template <typename Request, typename Response>
bool ClientSingleUpMultipleDown<Request, Response>::StateReadDone(bool ok)
{
    if (!ok)
    {
        DVLOG(1) << "server closed the read/download portion of the stream";
        m_NextState = &Client::StateFinishDone;
        m_Stream->Finish(&m_Status, this->Tag());
        return true;
    }

    DVLOG(1) << "issuing callback on received message";
    CallbackOnResponseReceived(std::move(m_Response));

    DVLOG(2) << "issuing next read from stream";
    m_NextState = &Client::StateReadDone;
    m_Stream->Read(&m_Response, this->Tag());
}

template <typename Request, typename Response>
bool ClientSingleUpMultipleDown<Request, Response>::StateFinishDone(bool ok)
{
    if (!ok)
    {
        DVLOG(1) << "failed to close the read/download portion of the stream";
        m_Context.TryCancel();
        return false;
        m_NextState = &Client::StateInvalid;
    }

    DVLOG(1) << "calling on complete callback";
    CallbackOnComplete(m_Status);

    m_NextState = &Client::StateInvalid;
    return false;
}

template <typename Request, typename Response>
bool ClientSingleUpMultipleDown<Request, Response>::StateInvalid(bool ok)
{
    LOG(FATAL) << "logic error in ClientSingleUpMultipleDown state management";
    return false;
}

template <typename Request, typename Response>
class ClientSUMD : ClientSingleUpMultipleDown<Request, Response>
{
    using Client = ClientSingleUpMultipleDown<Request, Response>;

  public:
    using PrepareFn            = typename Client::PrepareFn;
    using MetaData             = std::multimap<::grpc::string_ref, ::grpc::string_ref>;
    using CallbackOnResponseFn = std::function<void(Response&&)>;
    using CallbackOnCompleteFn = std::function<void(const ::grpc::Status&, MetaData&)>;

    ClientSUMD(PrepareFn prepare_fn,
               std::shared_ptr<Executor> executor,
               CallbackOnResponseFn callback,
               CallbackOnCompleteFn completer) :
      Client(prepare_fn, executor),
      m_Callback(callback),
      m_Completer(completer)
    {}

    std::shared_future<::grpc::Status> Write(Request&& request)
    {
        Client::Write(std::move(request));
        return Status();
    }

    std::shared_future<::grpc::Status> Status()
    {
        return m_Promise.get_future().share();
    }

  private:
    std::promise<::grpc::Status> m_Promise;
    CallbackOnResponseFn m_Callback;
    CallbackOnCompleteFn m_Completer;

    void CallbackOnResponseReceived(Response&& response) final override
    {
        m_Callback(std::move(response));
    }

    void CallbackOnComplete(const ::grpc::Status& status) final override
    {
        m_Promise.set_value(status);
        auto metadata = this->GetClientContext().GetServerTrailingMetadata();
        m_Completer(status, metadata);
    }
};

}  // namespace client
}  // namespace nvrpc

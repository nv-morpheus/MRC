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

#include <nvrpc/interfaces.h>

namespace nvrpc {

template <class Request, class Response>
class LifeCycleUnary : public IContextLifeCycle
{
  public:
    using RequestType           = Request;
    using ResponseType          = Response;
    using ServiceQueueFuncType  = std::function<void(::grpc::ServerContext*,
                                                    RequestType*,
                                                    ::grpc::ServerAsyncResponseWriter<ResponseType>*,
                                                    ::grpc::CompletionQueue*,
                                                    ::grpc::ServerCompletionQueue*,
                                                    void*)>;
    using ExecutorQueueFuncType = std::function<void(
        ::grpc::ServerContext*, RequestType*, ::grpc::ServerAsyncResponseWriter<ResponseType>*, void*)>;

    ~LifeCycleUnary() override {}

  protected:
    LifeCycleUnary() = default;
    void SetQueueFunc(ExecutorQueueFuncType);

    virtual void ExecuteRPC(RequestType& request, ResponseType& response) = 0;

    void FinishResponse() final override;
    void CancelResponse() final override;

    bool CheckDeadlineAndShouldContinue();

    const std::multimap<grpc::string_ref, grpc::string_ref>& ClientMetadata();

  private:
    // IContext Methods
    bool RunNextState(bool ok) final override;
    void Reset() final override;

    // LifeCycleUnary Specific Methods
    bool StateRequestDone(bool ok);
    bool StateFinishedDone(bool ok);

    // Function pointers
    ExecutorQueueFuncType m_QueuingFunc;
    bool (LifeCycleUnary<RequestType, ResponseType>::*m_NextState)(bool);

    // Variables
    std::unique_ptr<RequestType> m_Request;
    std::unique_ptr<ResponseType> m_Response;
    std::unique_ptr<::grpc::ServerContext> m_Context;
    std::unique_ptr<::grpc::ServerAsyncResponseWriter<ResponseType>> m_ResponseWriter;

  public:
    template <class RequestFuncType, class ServiceType>
    static ServiceQueueFuncType BindServiceQueueFunc(
        /*
        std::function<void(
            ServiceType *, ::grpc::ServerContext *, RequestType *,
            ::grpc::ServerAsyncResponseWriter<ResponseType> *,
            ::grpc::CompletionQueue *, ::grpc::ServerCompletionQueue *, void *)>
        */
        RequestFuncType request_fn,
        ServiceType* service_type)
    {
        return std::bind(request_fn,
                         service_type,
                         std::placeholders::_1,  // ServerContext*
                         std::placeholders::_2,  // InputType
                         std::placeholders::_3,  // AsyncResponseWriter<OutputType>
                         std::placeholders::_4,  // CQ
                         std::placeholders::_5,  // ServerCQ
                         std::placeholders::_6   // Tag
        );
    }

    static ExecutorQueueFuncType BindExecutorQueueFunc(ServiceQueueFuncType service_q_fn,
                                                       ::grpc::ServerCompletionQueue* cq)
    {
        return std::bind(service_q_fn,
                         std::placeholders::_1,  // ServerContext*
                         std::placeholders::_2,  // Request *
                         std::placeholders::_3,  // AsyncResponseWriter<Response> *
                         cq,
                         cq,
                         std::placeholders::_4  // Tag
        );
    }
};

// Implementation

template <class Request, class Response>
bool LifeCycleUnary<Request, Response>::RunNextState(bool ok)
{
    return (this->*m_NextState)(ok);
}

template <class Request, class Response>
void LifeCycleUnary<Request, Response>::Reset()
{
    OnLifeCycleReset();
    m_Request.reset(new Request);
    m_Response.reset(new Response);
    m_Context.reset(new ::grpc::ServerContext);
    m_ResponseWriter.reset(new ::grpc::ServerAsyncResponseWriter<ResponseType>(m_Context.get()));
    m_NextState = &LifeCycleUnary<RequestType, ResponseType>::StateRequestDone;
    m_QueuingFunc(m_Context.get(), m_Request.get(), m_ResponseWriter.get(), IContext::Tag());
}

template <class Request, class Response>
const std::multimap<grpc::string_ref, grpc::string_ref>& LifeCycleUnary<Request, Response>::ClientMetadata()
{
    return m_Context->client_metadata();
}

template <class Request, class Response>
bool LifeCycleUnary<Request, Response>::CheckDeadlineAndShouldContinue()
{
    // this segfaults
    // todo: work on the check deadline feature
    /*
    if(m_Context->IsCancelled())
    {
        CancelResponse();
        return false;
    }
    */
    return true;
}

template <class Request, class Response>
bool LifeCycleUnary<Request, Response>::StateRequestDone(bool ok)
{
    if (!ok)
    {
        return false;
    }
    OnLifeCycleStart();
    if (CheckDeadlineAndShouldContinue())
    {
        ExecuteRPC(*m_Request, *m_Response);
    }
    return true;
}

template <class Request, class Response>
bool LifeCycleUnary<Request, Response>::StateFinishedDone(bool ok)
{
    return false;
}

template <class Request, class Response>
void LifeCycleUnary<Request, Response>::FinishResponse()
{
    m_NextState = &LifeCycleUnary<RequestType, ResponseType>::StateFinishedDone;
    m_ResponseWriter->Finish(*m_Response, ::grpc::Status::OK, IContext::Tag());
}

template <class Request, class Response>
void LifeCycleUnary<Request, Response>::CancelResponse()
{
    m_NextState = &LifeCycleUnary<RequestType, ResponseType>::StateFinishedDone;
    m_ResponseWriter->Finish(*m_Response, ::grpc::Status::CANCELLED, IContext::Tag());
}

template <class Request, class Response>
void LifeCycleUnary<Request, Response>::SetQueueFunc(ExecutorQueueFuncType queue_fn)
{
    m_QueuingFunc = queue_fn;
}

}  // namespace nvrpc

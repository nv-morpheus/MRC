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

#include <nvrpc/client/base_context.h>
#include <nvrpc/client/executor.h>

#include <srf/core/async_compute.h>

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include <functional>
#include <memory>

namespace nvrpc {
namespace client {
namespace v2 {
template <typename Request, typename Response>
struct ClientStreaming : public BaseContext
{
  public:
    using PrepareFn = std::function<std::unique_ptr<::grpc::ClientAsyncReaderWriter<Request, Response>>(
        ::grpc::ClientContext*, ::grpc::CompletionQueue*)>;

    ClientStreaming(PrepareFn, std::shared_ptr<Executor>, bool initialize = true);
    ~ClientStreaming() override {}

    // void Write(Request*);
    bool Write(Request&&);

    std::shared_future<::grpc::Status> Status();
    std::shared_future<::grpc::Status> Done();
    std::shared_future<::grpc::Status> Cancel();

    bool SetCorked(bool true_or_false)
    {
        m_Corked = true_or_false;
    }

    bool IsCorked() const
    {
        return m_Corked;
    }

    bool ExecutorShouldDeleteContext() const override
    {
        return false;
    }

    void ExecutorShouldDeleteContext(bool true_or_false)
    {
        m_ShouldDelete = true_or_false;
    }

  protected:
    void Initialize();

    ::grpc::ClientContext& GetClientContext()
    {
        return m_Context;
    }

  private:
    bool RunNextState(bool ok) final override
    {
        return (this->*m_NextState)(ok);
    }

    bool RunNextState(bool (ClientStreaming<Request, Response>::*state_fn)(bool), bool ok)
    {
        return (this->*state_fn)(ok);
    }

    class Context : public BaseContext
    {
      public:
        Context(BaseContext* master) : BaseContext(master) {}
        ~Context() override {}

      private:
        bool RunNextState(bool ok) final override
        {
            // DVLOG(1) << this << ": " << "Event for Tag: " << Tag();
            return static_cast<ClientStreaming*>(m_MasterContext)->RunNextState(m_NextState, ok);
        }

        bool (ClientStreaming<Request, Response>::*m_NextState)(bool);

        bool ExecutorShouldDeleteContext() const override
        {
            return false;
        }

        friend class ClientStreaming<Request, Response>;
    };

    ::grpc::Status m_Status;
    ::grpc::ClientContext m_Context;
    std::unique_ptr<::grpc::ClientAsyncReaderWriter<Request, Response>> m_Stream;
    std::promise<::grpc::Status> m_Promise;
    std::shared_future<::grpc::Status> m_SharedFuture;

    PrepareFn m_PrepareFn;

    virtual void CallbackOnInitialized() {}
    virtual void CallbackOnRequestSent(Request&&) {}
    virtual void CallbackOnResponseReceived(Response&&) = 0;
    virtual void CallbackOnComplete(const ::grpc::Status&) {}

    Context m_ReadState;
    Context m_WriteState;

    std::mutex m_Mutex;
    std::queue<Response> m_ReadQueue;
    std::queue<Request> m_WriteQueue;

    std::shared_ptr<Executor> m_Executor;

    bool m_Corked;
    bool m_ShouldDelete;
    bool m_Initialized;

    using ReadHandle     = bool;
    using WriteHandle    = bool;
    using CloseHandle    = bool;
    using FinishHandle   = bool;
    using CompleteHandle = bool;
    using Actions        = std::tuple<ReadHandle, WriteHandle, CloseHandle, FinishHandle, CompleteHandle>;

    bool m_Reading, m_Writing, m_Finishing, m_Closing, m_ReadsDone, m_WritesDone, m_ClosedDone, m_FinishDone,
        m_Complete;

    bool (ClientStreaming<Request, Response>::*m_NextState)(bool);

    Actions EvaluateState();
    void ForwardProgress(Actions& actions);

    bool StateStreamInitialized(bool);
    bool StateReadDone(bool);
    bool StateWriteDone(bool);
    bool StateWritesDoneDone(bool);
    bool StateFinishDone(bool);
    bool StateInvalid(bool);
    bool StateIdle(bool);
};

template <typename Request, typename Response>
ClientStreaming<Request, Response>::ClientStreaming(PrepareFn prepare_fn,
                                                    std::shared_ptr<Executor> executor,
                                                    bool initialize) :
  m_Executor(executor),
  m_PrepareFn(prepare_fn),
  m_ReadState(this),
  m_WriteState(this),
  m_Reading(false),
  m_Writing(false),
  m_Finishing(false),
  m_Closing(false),
  m_ReadsDone(false),
  m_WritesDone(false),
  m_ClosedDone(false),
  m_FinishDone(false),
  m_Complete(false),
  m_ShouldDelete(false),
  m_Corked(false),
  m_Initialized(false)
{
    m_NextState              = &ClientStreaming<Request, Response>::StateInvalid;
    m_ReadState.m_NextState  = &ClientStreaming<Request, Response>::StateInvalid;
    m_WriteState.m_NextState = &ClientStreaming<Request, Response>::StateInvalid;

    Initialize();

    m_SharedFuture = m_Promise.get_future().share();
}

template <typename Request, typename Response>
void ClientStreaming<Request, Response>::Initialize()
{
    std::lock_guard<std::mutex> lock(m_Mutex);

    m_Initialized = true;

    m_NextState = &ClientStreaming<Request, Response>::StateStreamInitialized;
    m_Stream    = m_PrepareFn(&m_Context, m_Executor->GetNextCQ());
    m_Stream->StartCall(this->Tag());
}

template <typename Request, typename Response>
bool ClientStreaming<Request, Response>::Write(Request&& request)
{
    Actions actions;
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        CHECK(m_Initialized);

        if (m_WritesDone)
        {
            LOG(WARNING) << this << ": Attempting to Write on a Stream that is already closed";
            return false;
        }

        DVLOG(1) << this << ": Queuing Write Request";
        DVLOG(3) << this << ": On Queuing Write, there are " << m_WriteQueue.size() << " outstanding writes";

        m_WriteQueue.push(std::move(request));
        actions = EvaluateState();
    }
    ForwardProgress(actions);
    return true;
}

template <typename Request, typename Response>
std::shared_future<::grpc::Status> ClientStreaming<Request, Response>::Done()
{
    Actions actions;
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        CHECK(m_Initialized);

        if (m_WritesDone)
        {
            LOG(WARNING) << this << ": Attempting to Close (issue WritesDone) to a Stream that is already closed";
            return m_SharedFuture;
        }

        DVLOG(1) << this << ": Queuing WritesDone - no more Writes can be queued";
        DVLOG(3) << this << ": On Queuing WritesDone, there are " << m_WriteQueue.size() << " outstanding writes";

        m_WritesDone = true;
        actions      = EvaluateState();
    }
    ForwardProgress(actions);
    return m_SharedFuture;
}

template <typename Request, typename Response>
std::shared_future<::grpc::Status> ClientStreaming<Request, Response>::Cancel()
{
    Actions actions;
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        DVLOG(1) << this << ": Client Cancelling Stream";
        DVLOG(3) << this << ": On Cancel, there are " << m_WriteQueue.size() << " outstanding writes";

        CHECK(m_Initialized);

        m_WritesDone = true;
        m_Context.TryCancel();
    }
    return m_SharedFuture;
}

template <typename Request, typename Response>
std::shared_future<::grpc::Status> ClientStreaming<Request, Response>::Status()
{
    return m_SharedFuture;
}

template <typename Request, typename Response>
typename ClientStreaming<Request, Response>::Actions ClientStreaming<Request, Response>::EvaluateState()
{
    ReadHandle should_read         = false;
    WriteHandle should_write       = nullptr;
    CloseHandle should_close       = false;
    FinishHandle should_finish     = false;
    CompleteHandle should_complete = false;

    if (m_NextState == &ClientStreaming<Request, Response>::StateStreamInitialized)
    {
        DVLOG(1) << this << ": Action Queued: Stream Initializing";
    }
    else
    {
        if (!m_Reading && !m_ReadsDone)
        {
            should_read = true;
            m_Reading   = true;
            m_ReadQueue.emplace();
            m_ReadState.m_NextState = &ClientStreaming<Request, Response>::StateReadDone;
        }
        if (!m_Writing && !m_WriteQueue.empty())
        {
            should_write             = true;
            m_Writing                = true;
            m_WriteState.m_NextState = &ClientStreaming<Request, Response>::StateWriteDone;
            DVLOG(3) << this << ": WriteQueue has " << m_WriteQueue.size() << " outstanding messages";
        }
        if (!m_Closing && !m_Writing && m_WritesDone)
        {
            should_close             = true;
            m_Closing                = true;
            m_WriteState.m_NextState = &ClientStreaming<Request, Response>::StateWritesDoneDone;
        }
        if (!m_Reading && !m_Writing && !m_Finishing && m_ReadsDone && m_WritesDone && m_ClosedDone && !m_FinishDone)
        {
            should_finish = true;
            m_Finishing   = true;
            m_NextState   = &ClientStreaming<Request, Response>::StateFinishDone;
            DCHECK((m_ReadState.m_NextState == &ClientStreaming<Request, Response>::StateInvalid));
            DCHECK((m_WriteState.m_NextState == &ClientStreaming<Request, Response>::StateInvalid));
        }
        if (m_ReadsDone && m_WritesDone && m_ClosedDone && m_FinishDone && !m_Complete)
        {
            m_Complete      = true;
            should_complete = true;
        }
    }

    // clang-format off
                DVLOG(1) << this << ": " << (should_read ? 1 : 0) << (should_write ? 1 : 0) << (should_finish ? 1 : 0)
                    << " -- " << m_Reading << m_Writing << m_Finishing
                    << " -- " << m_ReadsDone << m_WritesDone
                    << " -- " << m_Finishing;
    // clang-format on

    return std::make_tuple(should_read, should_write, should_close, should_finish, should_complete);
}

template <class Request, class Response>
void ClientStreaming<Request, Response>::ForwardProgress(Actions& actions)
{
    ReadHandle should_read         = std::get<0>(actions);
    WriteHandle should_write       = std::get<1>(actions);
    CloseHandle should_close       = std::get<2>(actions);
    FinishHandle should_finish     = std::get<3>(actions);
    CompleteHandle should_complete = std::get<4>(actions);

    if (should_read)
    {
        DVLOG(1) << this << ": Posting Read/Recv";
        m_Stream->Read(&m_ReadQueue.back(), m_ReadState.Tag());
    }
    if (should_write)
    {
        DVLOG(1) << this << ": Writing/Sending Request";
        if (m_Corked)
        {
            ::grpc::WriteOptions options;
            options.set_corked();
            m_Stream->Write(m_WriteQueue.front(), options, m_WriteState.Tag());
        }
        else
        {
            m_Stream->Write(m_WriteQueue.front(), m_WriteState.Tag());
        }
    }
    if (should_close)
    {
        DVLOG(1) << this << ": Sending WritesDone to Server";
        m_Stream->WritesDone(m_WriteState.Tag());
    }
    if (should_finish)
    {
        DVLOG(1) << this << ": Closing Stream - Finish";
        m_Stream->Finish(&m_Status, Tag());
    }
    if (should_complete)
    {
        DVLOG(1) << this << ": Completing Promise";
        CallbackOnComplete(m_Status);
        m_Promise.set_value(std::move(m_Status));
    }
}

template <typename Request, typename Response>
bool ClientStreaming<Request, Response>::StateStreamInitialized(bool ok)
{
    if (!ok)
    {
        LOG(ERROR) << this << ": Stream Failed to Initialize";
        m_Status    = ::grpc::Status::CANCELLED;
        m_ReadsDone = m_WritesDone = m_ClosedDone = m_FinishDone = m_Complete = true;
        m_Promise.set_value(::grpc::Status::CANCELLED);
        CallbackOnComplete(m_Status);
        return false;
    }

    Actions actions;
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        DVLOG(1) << this << ": StreamInitialized";

        m_NextState = &ClientStreaming<Request, Response>::StateInvalid;
        actions     = EvaluateState();
    }

    CallbackOnInitialized();
    ForwardProgress(actions);
    return true;
}

template <typename Request, typename Response>
bool ClientStreaming<Request, Response>::StateReadDone(bool ok)
{
    Actions actions;
    std::function<void()> callback = nullptr;

    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        DVLOG(1) << this << ": ReadDone: " << (ok ? "OK" : "NOT OK");

        m_Reading               = false;
        m_ReadState.m_NextState = &ClientStreaming<Request, Response>::StateInvalid;

        if (!ok)
        {
            DVLOG(1) << this << ": Server is closing the read/download portion of the stream";
            m_ReadsDone  = true;
            m_WritesDone = true;
            if (m_Writing)
            {
                DLOG(WARNING) << "ReadDone with NOT OK; however, there is still an outstanding Write";
                m_Context.TryCancel();
            }
        }
        else
        {
            callback = [this, response = std::move(m_ReadQueue.front())]() mutable {
                CallbackOnResponseReceived(std::move(response));
            };
            m_ReadQueue.pop();
        }
        actions = EvaluateState();
    }

    // drop mutex and perform actions
    if (callback)
    {
        callback();
    }
    ForwardProgress(actions);
    return true;
}

template <typename Request, typename Response>
bool ClientStreaming<Request, Response>::StateWriteDone(bool ok)
{
    Actions actions;
    std::function<void()> callback = nullptr;

    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        DVLOG(1) << this << ": WriteDone: " << (ok ? "OK" : "NOT OK");

        m_Writing                = false;
        m_WriteState.m_NextState = &ClientStreaming<Request, Response>::StateInvalid;

        if (!ok)
        {
            // Invalidate any outstanding reads on stream
            DLOG(ERROR) << "Failed to Write to Stream - shutting down";
            m_WritesDone = true;
            m_WriteQueue = std::queue<Request>();
            if (!m_ReadsDone)
            {
                m_Context.TryCancel();
            }
        }
        else
        {
            callback = [this, request = std::move(m_WriteQueue.front())]() mutable {
                CallbackOnRequestSent(std::move(request));
            };
            m_WriteQueue.pop();
        }

        actions = EvaluateState();
    }

    // drop mutex and perform any actions
    if (callback)
    {
        callback();
    }
    ForwardProgress(actions);
    return true;
}

template <typename Request, typename Response>
bool ClientStreaming<Request, Response>::StateWritesDoneDone(bool ok)
{
    Actions actions;
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        DVLOG(1) << this << ": WritesDoneDone: " << (ok ? "OK" : "NOT OK");

        m_ClosedDone             = true;
        m_WriteState.m_NextState = &ClientStreaming<Request, Response>::StateInvalid;

        if (!ok)
        {
            LOG(ERROR) << "Failed to close write/upload portion of stream";
            if (!m_ReadsDone)
            {
                m_Context.TryCancel();
            }
        }

        actions = EvaluateState();
    }
    ForwardProgress(actions);
    return true;
}

template <typename Request, typename Response>
bool ClientStreaming<Request, Response>::StateFinishDone(bool ok)
{
    Actions actions;
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        DVLOG(1) << this << ": FinishedDone: " << (ok ? "OK" : "NOT OK");

        m_Finishing  = false;
        m_FinishDone = true;

        if (!ok)
        {
            LOG(ERROR) << "Request to Finish the stream failed";
            m_Context.TryCancel();
        }

        actions = EvaluateState();
    }
    ForwardProgress(actions);
    return false;
}

template <typename Request, typename Response>
bool ClientStreaming<Request, Response>::StateInvalid(bool ok)
{
    LOG(FATAL) << "Your logic is bad - you should never have come here";
}

}  // namespace v2
}  // namespace client
}  // namespace nvrpc

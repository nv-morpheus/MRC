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

#include <nvrpc/executor.h>

#include <nvrpc/interfaces.h>
#include <nvrpc/thread_pool.h>

#include <glog/logging.h>
#include <grpc/support/time.h>

#include <utility>  // for move

using ::nvrpc::ThreadPool;

namespace nvrpc {

Worker::Worker(std::size_t cq_count) : m_cq_count(cq_count)
{
    m_TimeoutCallback = [] {};
}

void Worker::Initialize(::grpc::ServerBuilder& builder)
{
    for (int i = 0; i < m_cq_count; i++)
    {
        m_ServerCompletionQueues.emplace_back(builder.AddCompletionQueue());
    }
}

void Worker::Shutdown()
{
    if (!m_Running)
        return;

    m_Running = false;

    for (auto& cq : m_ServerCompletionQueues)
    {
        VLOG(3) << "Telling CQ to Shutdown: " << cq.get();
        cq->Shutdown();
    }
}

void Worker::ProgressEngine(int thread_id)
{
    bool ok;
    void* tag;
    std::uint64_t backoff = 1;

    auto myCQ           = m_ServerCompletionQueues[thread_id].get();
    using NextStatus    = ::grpc::ServerCompletionQueue::NextStatus;
    m_Running           = true;
    const bool is_async = IsAsync();

    gpr_timespec timespec;
    timespec.clock_type = GPR_TIMESPAN;
    timespec.tv_sec     = 10;
    timespec.tv_nsec    = 0;

    DVLOG(10) << "starting progress engine";

    while (true)
    {
        switch (myCQ->AsyncNext<gpr_timespec>(&tag, &ok, (is_async ? gpr_time_0(GPR_CLOCK_REALTIME) : timespec)))
        {
        case NextStatus::GOT_EVENT: {
            backoff  = 1;
            auto ctx = IContext::Detag(tag);
            if (!RunContext(ctx, ok))
            {
                if (m_Running)
                {
                    ResetContext(ctx);
                }
            }
        }
        break;
        case grpc::CompletionQueue::NextStatus::TIMEOUT: {
            if (backoff < 1048576)
            {
                backoff = (backoff << 1);
            }
            TimeoutBackoff(backoff);
        }
        break;
        case grpc::CompletionQueue::NextStatus::SHUTDOWN: {
            DVLOG(10) << "progress engine complete";
            return;
        }
        }
    }
}

void Worker::SetTimeout(time_point deadline, std::function<void()> callback)
{
    m_TimeoutDeadline = deadline;
    m_TimeoutCallback = callback;
}

IContext& Worker::context(std::size_t idx)
{
    CHECK_LT(idx, m_Contexts.size());
    return *(m_Contexts[idx]);
}

std::size_t Worker::context_count() const
{
    return m_Contexts.size();
}

Executor::Executor(std::size_t numThreads) : Worker(numThreads), m_ThreadPool(std::make_unique<ThreadPool>(numThreads))
{}

Executor::Executor(std::unique_ptr<nvrpc::ThreadPool> threadpool) : m_ThreadPool(std::move(threadpool)) {}

}  // namespace nvrpc

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

#include "nvrpc/client/executor.h"  // IWYU pragma: associated

#include "nvrpc/client/base_context.h"

#include <nvrpc/thread_pool.h>

#include <glog/logging.h>

#include <future>   // for m_ThreadPool->enqueue
#include <ostream>  // for logging ostream<<
#include <utility>  // for move

using ::nvrpc::ThreadPool;

namespace nvrpc::client {

Executor::Executor() : Executor(1) {}

Executor::Executor(int numThreads) : Executor(std::make_unique<ThreadPool>(numThreads)) {}

Executor::Executor(std::unique_ptr<ThreadPool> threadpool) : m_ThreadPool(std::move(threadpool)), m_Counter(0)
{
    // for(decltype(m_ThreadPool->Size()) i = 0; i < m_ThreadPool->Size(); i++)
    for (auto i = 0; i < m_ThreadPool->size(); i++)
    {
        DVLOG(1) << "Starting Client Progress Engine #" << i;
        m_CQs.emplace_back(new ::grpc::CompletionQueue);
        auto cq = m_CQs.back().get();
        m_ThreadPool->enqueue([this, cq] { ProgressEngine(*cq); });
    }
}

Executor::~Executor()
{
    ShutdownAndJoin();
}

void Executor::ShutdownAndJoin()
{
    for (auto& cq : m_CQs)
    {
        cq->Shutdown();
    }
    m_ThreadPool.reset();
}

void Executor::ProgressEngine(::grpc::CompletionQueue& cq)
{
    void* tag;
    bool ok = false;

    while (cq.Next(&tag, &ok))
    {
        // CHECK(ok);
        BaseContext* ctx = BaseContext::Detag(tag);
        DVLOG(3) << "executor issuing callback";
        auto should_delete = ctx->ExecutorShouldDeleteContext();
        if (!ctx->RunNextState(ok))
        {
            if (should_delete)
            {
                DVLOG(1) << "Deleting ClientContext: " << tag;
                delete ctx;
            }
        }
        DVLOG(3) << "executor callback complete";
    }
}

::grpc::CompletionQueue* Executor::GetNextCQ() const
{
    std::size_t idx = 0;
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        DCHECK_LT(m_Counter, m_ThreadPool->size());
        if (++m_Counter == m_ThreadPool->size())
        {
            m_Counter = 0;
        }
        idx = m_Counter;
    }
    return m_CQs[idx].get();
}

}  // namespace nvrpc::client

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

#include "nvrpc/interfaces.h"
#include "nvrpc/thread_pool.h"

#include <glog/logging.h>

#include <grpcpp/grpcpp.h>

#include <cstddef>  // for size_t
#include <cstdint>  // for uint64_t
#include <functional>
#include <future>   // for m_ThreadPool->enqueue
#include <memory>   // for unique_ptr, shared_ptr
#include <ostream>  // for logging ostream<<
#include <vector>
// work-around for known iwyu issue usage of vector implies usage of std::max
// https://github.com/include-what-you-use/include-what-you-use/issues/908
// IWYU pragma: no_include <algorithm>

namespace nvrpc {

class Worker : public IExecutor
{
  public:
    Worker(std::size_t cq_count = 1);
    ~Worker() override = default;

    void Initialize(::grpc::ServerBuilder& builder) final;
    void RegisterContexts(IRPC* rpc, std::shared_ptr<Resources> resources, int numContextsPerThread) final
    {
        auto base = dynamic_cast<IExecutor*>(this);
        for (int i = 0; i < m_ServerCompletionQueues.size(); i++)
        {
            auto cq = m_ServerCompletionQueues[i].get();
            for (int j = 0; j < numContextsPerThread; j++)
            {
                DVLOG(20) << "Creating Context " << j << " on CQ " << i;
                m_Contexts.emplace_back(this->CreateContext(rpc, cq, resources));
            }
        }
    }

    void Shutdown() override;

    std::size_t Size() const
    {
        return m_cq_count;
    }

    bool IsRunning() const
    {
        return m_Running;
    }

  protected:
    virtual void ProgressEngine(int thread_id);

    virtual bool IsAsync() const
    {
        return false;
    }

    virtual void TimeoutBackoff(std::uint64_t) {}

    void SetTimeout(time_point, std::function<void()>) override;

    IContext& context(std::size_t i);
    std::size_t context_count() const;

  private:
    std::size_t m_cq_count{1};
    volatile bool m_Running{false};
    time_point m_TimeoutDeadline;
    std::function<void()> m_TimeoutCallback{nullptr};
    std::vector<std::unique_ptr<IContext>> m_Contexts;
    std::vector<std::unique_ptr<::grpc::ServerCompletionQueue>> m_ServerCompletionQueues;
};

class Executor : public Worker
{
  public:
    Executor(std::size_t numThreads = 1);
    Executor(std::unique_ptr<::nvrpc::ThreadPool> threadpool);
    ~Executor() override = default;

    void Run() final
    {
        // Launch the threads polling on their CQs
        for (int i = 0; i < m_ThreadPool->size(); i++)
        {
            m_ThreadPool->enqueue([this, i] { ProgressEngine(i); });
        }
        // Queue the Execution Contexts in the recieve queue
        for (int i = 0; i < context_count(); i++)
        {
            ResetContext(&context(i));
            // TODO: add a hooked to allow one to customize some
            // ContextDidReset(m_Contexts[i].get());
        }
    }

  protected:
    std::unique_ptr<::nvrpc::ThreadPool> m_ThreadPool;
};

}  // namespace nvrpc

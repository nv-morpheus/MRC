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

#include <nvrpc/thread_pool.h>

#include <grpcpp/grpcpp.h>  // for ServerCompletionQueue

#include <cstddef>  // for size_t
#include <memory>
#include <mutex>
#include <vector>

namespace nvrpc::client {

class Executor : public std::enable_shared_from_this<Executor>
{
  public:
    Executor();
    Executor(int numThreads);
    Executor(std::unique_ptr<::nvrpc::ThreadPool> threadpool);

    Executor(Executor&& other) noexcept = delete;
    Executor& operator=(Executor&& other) noexcept = delete;

    Executor(const Executor& other) = delete;
    Executor& operator=(const Executor& other) = delete;

    virtual ~Executor();

    void ShutdownAndJoin();
    ::grpc::CompletionQueue* GetNextCQ() const;

  private:
    void ProgressEngine(::grpc::CompletionQueue&);

    mutable std::size_t m_Counter;
    std::unique_ptr<::nvrpc::ThreadPool> m_ThreadPool;
    std::vector<std::unique_ptr<::grpc::CompletionQueue>> m_CQs;
    mutable std::mutex m_Mutex;
};

}  // namespace nvrpc::client

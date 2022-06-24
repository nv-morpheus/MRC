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

#include "testing.grpc.pb.h"  // IWYU pragma: keep
#include "testing.pb.h"

#include <nvrpc/interfaces.h>
#include <nvrpc/life_cycle_streaming.h>
#include <nvrpc/thread_pool.h>

#include <cstddef>  // for size_t
#include <map>      // for map
#include <memory>   // for shared_ptr
#include <mutex>    // for mutex
// work-around for known iwyu issue
// https://github.com/include-what-you-use/include-what-you-use/issues/908
// IWYU pragma: no_include <algorithm>

namespace nvrpc {
namespace testing {

struct TestResources : public Resources
{
    TestResources(int numThreadsInPool = 3);

    using Stream   = std::shared_ptr<LifeCycleStreaming<Input, Output>::ServerStream>;
    using StreamID = std::size_t;
    using Counter  = std::size_t;

    ::nvrpc::ThreadPool& AcquireThreadPool();

    void StreamManagerInit();
    void StreamManagerFini();
    void StreamManagerWorker();

    void IncrementStreamCount(Stream);
    void CloseStream(Stream);

  private:
    ::nvrpc::ThreadPool m_ThreadPool;

    bool m_Running;
    std::mutex m_MessageMutex;
    std::map<StreamID, Stream> m_Streams;
    std::map<StreamID, Counter> m_MessagesRecv;
    std::map<StreamID, Counter> m_MessagesSent;

    std::mutex m_ShutdownMutex;
    bool m_ClientRunning;
    bool m_ServerRunning;
};

}  // namespace testing
}  // namespace nvrpc

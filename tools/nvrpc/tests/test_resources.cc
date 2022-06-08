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

#include "test_resources.h"

#include <glog/logging.h>

#include <chrono>   // for microseconds
#include <future>   // for m_ThreadPool.enqueue
#include <ostream>  // for logging ostream<<
#include <thread>   // for sleep_for
#include <utility>  // for move

namespace nvrpc::testing {

TestResources::TestResources(int numThreadsInPool) : m_ThreadPool(numThreadsInPool) {}

::nvrpc::ThreadPool& TestResources::AcquireThreadPool()
{
    return m_ThreadPool;
}

void TestResources::StreamManagerInit()
{
    std::lock_guard<std::mutex> lock(m_MessageMutex);
    m_Running = true;
    m_ThreadPool.enqueue([this]() mutable { StreamManagerWorker(); });
}

void TestResources::StreamManagerFini()
{
    std::lock_guard<std::mutex> lock(m_MessageMutex);
    m_Running = false;
}

void TestResources::StreamManagerWorker()
{
    while (m_Running)
    {
        {
            std::lock_guard<std::mutex> lock(m_MessageMutex);
            for (auto& item : m_Streams)
            {
                LOG_FIRST_N(INFO, 10) << "Progress Engine";
                auto stream_id = item.first;
                auto& stream   = item.second;

                for (size_t i = m_MessagesSent[stream_id] + 1; i <= m_MessagesRecv[stream_id]; i++)
                {
                    DLOG(INFO) << "Writing: " << i;
                    Output output;
                    output.set_batch_id(i);
                    stream->WriteResponse(std::move(output));
                }

                m_MessagesSent[stream_id] = m_MessagesRecv[stream_id];
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void TestResources::CloseStream(Stream stream)
{
    std::lock_guard<std::mutex> lock(m_MessageMutex);
    auto stream_id = stream->StreamID();

    m_Streams.erase(stream_id);
    m_MessagesRecv.erase(stream_id);
    m_MessagesSent.erase(stream_id);

    DLOG(INFO) << "****** Client Closed ****** ";
    stream->FinishStream();
}

void TestResources::IncrementStreamCount(Stream stream)
{
    std::lock_guard<std::mutex> lock(m_MessageMutex);
    auto stream_id = stream->StreamID();
    auto search    = m_Streams.find(stream_id);
    if (search == m_Streams.end())
    {
        m_Streams[stream_id]      = stream;
        m_MessagesRecv[stream_id] = 1;
    }
    else
    {
        m_MessagesRecv[stream_id]++;
    }
}

}  // namespace nvrpc::testing

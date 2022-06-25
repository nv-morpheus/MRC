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

#include "test_build_server.h"  // IWYU pragma: associated
#include "test_pingpong.h"

#include <nvrpc/server.h>
#include <nvrpc/thread_pool.h>

#include <gtest/gtest.h>

#include <chrono>  // for milliseconds
#include <condition_variable>
#include <future>  // for m_BackgroundThreads->enqueue
#include <memory>  // for unique_ptr
#include <mutex>
// work-around for known iwyu issue
// https://github.com/include-what-you-use/include-what-you-use/issues/908
// IWYU pragma: no_include <algorithm>
// IWYU pragma: no_include <functional>
// IWYU pragma: no_include <vector>

using namespace nvrpc;
using namespace nvrpc::testing;

class ServerTest : public ::testing::Test
{
    void SetUp() override
    {
        m_BackgroundThreads = std::make_unique<::nvrpc::ThreadPool>(1);
        m_Server            = BuildServer<PingPongUnaryContext, PingPongStreamingContext>();
    }

    void TearDown() override
    {
        if (m_Server)
        {
            m_Server->Shutdown();
            m_Server.reset();
        }
        m_BackgroundThreads.reset();
    }

  protected:
    std::unique_ptr<Server> m_Server;
    std::unique_ptr<::nvrpc::ThreadPool> m_BackgroundThreads;
};

TEST_F(ServerTest, AsyncStartAndShutdown)
{
    EXPECT_FALSE(m_Server->Running());
    m_Server->AsyncStart();
    EXPECT_TRUE(m_Server->Running());
    m_Server->Shutdown();
    EXPECT_FALSE(m_Server->Running());
}

TEST_F(ServerTest, RunAndShutdown)
{
    bool running = false;
    std::mutex mutex;
    std::condition_variable condition;

    EXPECT_FALSE(m_Server->Running());
    m_BackgroundThreads->enqueue([&, this] {
        m_Server->Run(std::chrono::milliseconds(1), [&] {
            {
                std::lock_guard<std::mutex> lock(mutex);
                running = true;
            }
            condition.notify_all();
        });
    });
    {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&running] { return running; });
    }
    EXPECT_TRUE(m_Server->Running());
    m_Server->Shutdown();
    EXPECT_FALSE(m_Server->Running());
}

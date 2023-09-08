/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tests/common.hpp"

#include "internal/service.hpp"

#include "mrc/exceptions/runtime_error.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <atomic>

namespace mrc {

class SimpleService : public Service
{
  public:
    SimpleService(bool do_call_in_destructor = true) :
      Service("SimpleService"),
      m_do_call_in_destructor(do_call_in_destructor)
    {}

    ~SimpleService() override
    {
        if (m_do_call_in_destructor)
        {
            Service::call_in_destructor();
        }
    }

    size_t start_call_count() const
    {
        return m_start_call_count.load();
    }

    size_t stop_call_count() const
    {
        return m_stop_call_count.load();
    }

    size_t kill_call_count() const
    {
        return m_kill_call_count.load();
    }

    size_t await_live_call_count() const
    {
        return m_await_live_call_count.load();
    }

    size_t await_join_call_count() const
    {
        return m_await_join_call_count.load();
    }

    void set_start_callback(std::function<void()> callback)
    {
        m_start_callback = std::move(callback);
    }

    void set_stop_callback(std::function<void()> callback)
    {
        m_stop_callback = std::move(callback);
    }

    void set_kill_callback(std::function<void()> callback)
    {
        m_kill_callback = std::move(callback);
    }

    void set_await_live_callback(std::function<void()> callback)
    {
        m_await_live_callback = std::move(callback);
    }

    void set_await_join_callback(std::function<void()> callback)
    {
        m_await_join_callback = std::move(callback);
    }

  private:
    void do_service_start() final
    {
        if (m_start_callback)
        {
            m_start_callback();
        }

        m_start_call_count++;
    }

    void do_service_stop() final
    {
        if (m_stop_callback)
        {
            m_stop_callback();
        }

        m_stop_call_count++;
    }

    void do_service_kill() final
    {
        if (m_kill_callback)
        {
            m_kill_callback();
        }

        m_kill_call_count++;
    }

    void do_service_await_live() final
    {
        if (m_await_live_callback)
        {
            m_await_live_callback();
        }

        m_await_live_call_count++;
    }

    void do_service_await_join() final
    {
        if (m_await_join_callback)
        {
            m_await_join_callback();
        }

        m_await_join_call_count++;
    }

    bool m_do_call_in_destructor{true};

    std::atomic_size_t m_start_call_count{0};
    std::atomic_size_t m_stop_call_count{0};
    std::atomic_size_t m_kill_call_count{0};
    std::atomic_size_t m_await_live_call_count{0};
    std::atomic_size_t m_await_join_call_count{0};

    std::function<void()> m_start_callback;
    std::function<void()> m_stop_callback;
    std::function<void()> m_kill_callback;
    std::function<void()> m_await_live_callback;
    std::function<void()> m_await_join_callback;
};

class TestService : public ::testing::Test
{
  protected:
};

TEST_F(TestService, LifeCycle)
{
    SimpleService service;

    service.service_start();

    EXPECT_EQ(service.state(), ServiceState::Running);
    EXPECT_EQ(service.start_call_count(), 1);

    service.service_await_live();

    EXPECT_EQ(service.await_live_call_count(), 1);

    service.service_await_join();

    EXPECT_EQ(service.state(), ServiceState::Completed);
    EXPECT_EQ(service.await_join_call_count(), 1);

    EXPECT_EQ(service.stop_call_count(), 0);
    EXPECT_EQ(service.kill_call_count(), 0);
}

TEST_F(TestService, ServiceNotStarted)
{
    SimpleService service;

    EXPECT_ANY_THROW(service.service_await_live());
    EXPECT_ANY_THROW(service.service_stop());
    EXPECT_ANY_THROW(service.service_kill());
    EXPECT_ANY_THROW(service.service_await_join());
}

TEST_F(TestService, ServiceStop)
{
    SimpleService service;

    service.service_start();

    EXPECT_EQ(service.state(), ServiceState::Running);

    service.service_stop();

    EXPECT_EQ(service.state(), ServiceState::Stopping);

    service.service_await_join();

    EXPECT_EQ(service.state(), ServiceState::Completed);

    EXPECT_EQ(service.stop_call_count(), 1);
}

TEST_F(TestService, ServiceKill)
{
    SimpleService service;

    service.service_start();

    EXPECT_EQ(service.state(), ServiceState::Running);

    service.service_kill();

    EXPECT_EQ(service.state(), ServiceState::Killing);

    service.service_await_join();

    EXPECT_EQ(service.state(), ServiceState::Completed);

    EXPECT_EQ(service.kill_call_count(), 1);
}

TEST_F(TestService, ServiceStopThenKill)
{
    SimpleService service;

    service.service_start();

    EXPECT_EQ(service.state(), ServiceState::Running);

    service.service_stop();

    EXPECT_EQ(service.state(), ServiceState::Stopping);

    service.service_kill();

    EXPECT_EQ(service.state(), ServiceState::Killing);

    service.service_await_join();

    EXPECT_EQ(service.state(), ServiceState::Completed);

    EXPECT_EQ(service.stop_call_count(), 1);
    EXPECT_EQ(service.kill_call_count(), 1);
}

TEST_F(TestService, ServiceKillThenStop)
{
    SimpleService service;

    service.service_start();

    EXPECT_EQ(service.state(), ServiceState::Running);

    service.service_kill();

    EXPECT_EQ(service.state(), ServiceState::Killing);

    service.service_stop();

    EXPECT_EQ(service.state(), ServiceState::Killing);

    service.service_await_join();

    EXPECT_EQ(service.state(), ServiceState::Completed);

    EXPECT_EQ(service.stop_call_count(), 0);
    EXPECT_EQ(service.kill_call_count(), 1);
}

TEST_F(TestService, MultipleStartCalls)
{
    SimpleService service;

    service.service_start();

    // Call again (should be an error)
    EXPECT_ANY_THROW(service.service_start());

    EXPECT_EQ(service.start_call_count(), 1);
}

TEST_F(TestService, MultipleStopCalls)
{
    SimpleService service;

    service.service_start();

    // Multiple calls to stop are fine
    service.service_stop();
    service.service_stop();

    EXPECT_EQ(service.stop_call_count(), 1);
}

TEST_F(TestService, MultipleKillCalls)
{
    SimpleService service;

    service.service_start();

    // Multiple calls to kill are fine
    service.service_kill();
    service.service_kill();

    EXPECT_EQ(service.kill_call_count(), 1);
}

TEST_F(TestService, MultipleJoinCalls)
{
    SimpleService service;

    service.service_start();

    service.service_await_live();

    service.service_await_join();
    service.service_await_join();

    EXPECT_EQ(service.await_join_call_count(), 1);
}

TEST_F(TestService, StartWithException)
{
    SimpleService service;

    service.set_start_callback([]() {
        throw exceptions::MrcRuntimeError("Live Exception");
    });

    EXPECT_ANY_THROW(service.service_start());

    EXPECT_EQ(service.state(), ServiceState::Completed);
}

TEST_F(TestService, LiveWithException)
{
    SimpleService service;

    service.set_await_join_callback([]() {
        throw exceptions::MrcRuntimeError("Live Exception");
    });

    service.service_start();

    EXPECT_ANY_THROW(service.service_await_join());
}

TEST_F(TestService, MultipleLiveWithException)
{
    SimpleService service;

    service.set_await_live_callback([]() {
        throw exceptions::MrcRuntimeError("Live Exception");
    });

    service.service_start();

    EXPECT_ANY_THROW(service.service_await_live());
    EXPECT_ANY_THROW(service.service_await_live());
}

TEST_F(TestService, JoinWithException)
{
    SimpleService service;

    service.set_await_join_callback([]() {
        throw exceptions::MrcRuntimeError("Join Exception");
    });

    service.service_start();

    EXPECT_ANY_THROW(service.service_await_join());
}

TEST_F(TestService, MultipleJoinWithException)
{
    SimpleService service;

    service.set_await_join_callback([]() {
        throw exceptions::MrcRuntimeError("Join Exception");
    });

    service.service_start();

    EXPECT_ANY_THROW(service.service_await_join());
    EXPECT_ANY_THROW(service.service_await_join());
}

}  // namespace mrc

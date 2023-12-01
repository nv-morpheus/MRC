/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * Original Source: https://github.com/jbaldwin/libcoro
 * Original License: Apache License, Version 2.0; included below
 */

/**
 * Copyright 2021 Josh Baldwin
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mrc/core/expected.hpp"
#include "mrc/coroutines/ring_buffer.hpp"
#include "mrc/coroutines/sync_wait.hpp"
#include "mrc/coroutines/task.hpp"
#include "mrc/coroutines/thread_pool.hpp"
#include "mrc/coroutines/when_all.hpp"

#include <gtest/gtest.h>

#include <coroutine>
#include <cstdint>
#include <functional>
#include <memory>
#include <tuple>

using namespace mrc;

class TestCoroTask : public ::testing::Test
{};

static auto double_task = [](std::uint64_t x) -> coroutines::Task<std::uint64_t> {
    co_return x * 2;
};

static auto scheduled_task = [](coroutines::ThreadPool& tp, std::uint64_t x) -> coroutines::Task<std::uint64_t> {
    co_await tp.schedule();
    co_return x * 2;
};

static auto double_and_add_5_task = [](std::uint64_t input) -> coroutines::Task<std::uint64_t> {
    auto doubled = co_await double_task(input);
    co_return doubled + 5;
};

TEST_F(TestCoroTask, Task)
{
    auto output = coroutines::sync_wait(double_task(2));
    EXPECT_EQ(output, 4);
}

TEST_F(TestCoroTask, ScheduledTask)
{
    coroutines::ThreadPool main({.thread_count = 1, .description = "main"});
    auto output = coroutines::sync_wait(scheduled_task(main, 2));
    EXPECT_EQ(output, 4);
}

TEST_F(TestCoroTask, Tasks)
{
    auto output = coroutines::sync_wait(double_and_add_5_task(2));
    EXPECT_EQ(output, 9);
}

TEST_F(TestCoroTask, RingBufferStressTest)
{
    coroutines::ThreadPool writer({.thread_count = 1, .description = "writer"});
    coroutines::ThreadPool reader({.thread_count = 1, .description = "reader"});
    coroutines::RingBuffer<std::unique_ptr<std::uint64_t>> buffer({.capacity = 2});

    for (int iters = 16; iters <= 16; iters++)
    {
        auto source = [&writer, &buffer, iters]() -> coroutines::Task<void> {
            co_await writer.schedule();
            for (std::uint64_t i = 0; i < iters; i++)
            {
                co_await buffer.write(std::make_unique<std::uint64_t>(i));
            }
            co_return;
        };

        auto sink = [&reader, &buffer, iters]() -> coroutines::Task<void> {
            co_await reader.schedule();
            for (std::uint64_t i = 0; i < iters; i++)
            {
                auto unique = co_await buffer.read();
                EXPECT_TRUE(unique);
                EXPECT_EQ(*(unique.value()), i);
            }
            co_return;
        };

        coroutines::sync_wait(coroutines::when_all(source(), sink()));
    }
}

// this is our awaitable
class AwaitableTaskProvider
{
  public:
    struct Done
    {};

    AwaitableTaskProvider()
    {
        m_task_generator = []() -> coroutines::Task<mrc::expected<int, Done>> {
            co_return {42};
        };
    }

    auto operator co_await() -> decltype(auto)
    {
        return m_task_generator().operator co_await();
    }

  private:
    std::function<coroutines::Task<mrc::expected<int, Done>>()> m_task_generator;
};

TEST_F(TestCoroTask, AwaitableTaskProvider)
{
    auto expected = coroutines::sync_wait(AwaitableTaskProvider{});
    EXPECT_EQ(*expected, 42);

    auto task = []() -> coroutines::Task<void> {
        auto expected = co_await AwaitableTaskProvider{};
        EXPECT_EQ(*expected, 42);
        co_return;
    };

    coroutines::sync_wait(task());
}

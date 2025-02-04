/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/coroutines/async_generator.hpp"
#include "mrc/coroutines/io_scheduler.hpp"
#include "mrc/coroutines/sync_wait.hpp"
#include "mrc/coroutines/task.hpp"
#include "mrc/coroutines/time.hpp"
#include "mrc/coroutines/when_all.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <coroutine>
#include <cstdint>
#include <memory>
#include <ratio>
#include <utility>
#include <vector>

using namespace mrc;
using namespace std::chrono_literals;

class TestCoroIoScheduler : public ::testing::Test
{};

TEST_F(TestCoroIoScheduler, YieldFor)
{
    auto scheduler = coroutines::IoScheduler::get_instance();

    static constexpr std::chrono::milliseconds Delay{10};

    auto task = [scheduler]() -> coroutines::Task<> {
        co_await scheduler->yield_for(Delay);
    };

    auto start = coroutines::clock_t::now();
    coroutines::sync_wait(task());
    auto stop = coroutines::clock_t::now();

    ASSERT_GE(stop - start, Delay);
}

TEST_F(TestCoroIoScheduler, YieldUntil)
{
    auto scheduler = coroutines::IoScheduler::get_instance();

    coroutines::clock_t::time_point target_time{};

    auto task = [scheduler, &target_time]() -> coroutines::Task<> {
        target_time = coroutines::clock_t::now() + 10ms;
        co_await scheduler->yield_until(target_time);
    };

    coroutines::sync_wait(task());

    auto current_time = coroutines::clock_t::now();

    ASSERT_GE(current_time, target_time);
}

TEST_F(TestCoroIoScheduler, Concurrent)
{
    auto scheduler = coroutines::IoScheduler::get_instance();

    auto per_task_overhead = [&] {
        static constexpr std::chrono::milliseconds SmallestDelay{1};
        auto start = coroutines::clock_t::now();
        coroutines::sync_wait([scheduler]() -> coroutines::Task<> {
            co_await scheduler->yield_for(SmallestDelay);
        }());
        auto stop = coroutines::clock_t::now();
        return (stop - start) - SmallestDelay;
    }();

    static constexpr std::chrono::milliseconds TaskDuration{10};

    auto task = [scheduler]() -> coroutines::Task<> {
        co_await scheduler->yield_for(TaskDuration);
    };

    auto start = coroutines::clock_t::now();

    std::vector<coroutines::Task<>> tasks;

    const uint32_t NumTasks{1'000};
    for (uint32_t i = 0; i < NumTasks; i++)
    {
        tasks.push_back(task());
    }

    coroutines::sync_wait(coroutines::when_all(std::move(tasks)));
    auto stop = coroutines::clock_t::now();

    ASSERT_LT(stop - start, TaskDuration + per_task_overhead * NumTasks);
}

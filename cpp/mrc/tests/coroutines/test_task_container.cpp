/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/coroutines/sync_wait.hpp"
#include "mrc/coroutines/task.hpp"
#include "mrc/coroutines/task_container.hpp"
#include "mrc/coroutines/test_scheduler.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <coroutine>
#include <cstdint>
#include <memory>
#include <ratio>
#include <thread>
#include <vector>

class TestCoroTaskContainer : public ::testing::Test
{};

TEST_F(TestCoroTaskContainer, LifeCycle) {}

TEST_F(TestCoroTaskContainer, MaxSimultaneousTasks)
{
    using namespace std::chrono_literals;

    const int32_t num_threads          = 16;
    const int32_t num_tasks_per_thread = 16;
    const int32_t num_tasks            = num_threads * num_tasks_per_thread;
    const int32_t max_concurrent_tasks = 2;

    auto on             = std::make_shared<mrc::coroutines::TestScheduler>();
    auto task_container = mrc::coroutines::TaskContainer(on, max_concurrent_tasks);

    auto start_time = on->time();

    std::vector<std::chrono::time_point<std::chrono::steady_clock>> execution_times;

    auto delay = [](std::shared_ptr<mrc::coroutines::TestScheduler> on,
                    std::vector<std::chrono::time_point<std::chrono::steady_clock>>& execution_times)
        -> mrc::coroutines::Task<> {
        co_await on->yield_for(100ms);
        execution_times.emplace_back(on->time());
    };

    std::vector<std::thread> threads;

    for (auto i = 0; i < num_threads; i++)
    {
        threads.emplace_back([&]() {
            for (auto i = 0; i < num_tasks_per_thread; i++)
            {
                task_container.start(delay(on, execution_times));
            }
        });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    auto task = task_container.garbage_collect_and_yield_until_empty();

    task.resume();

    while (on->resume_next()) {}

    mrc::coroutines::sync_wait(task);

    ASSERT_EQ(execution_times.size(), num_tasks);

    for (auto i = 0; i < execution_times.size(); i++)
    {
        ASSERT_EQ(execution_times[i], start_time + (i / max_concurrent_tasks + 1) * 100ms) << "Failed at index " << i;
    }
}

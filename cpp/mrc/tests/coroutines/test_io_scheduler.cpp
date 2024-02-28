/**
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <utility>
#include <vector>

using namespace mrc;
using namespace std::chrono_literals;

class TestCoroIoScheduler : public ::testing::Test
{};

TEST_F(TestCoroIoScheduler, YieldFor)
{
    auto scheduler = coroutines::IoScheduler::get_instance();

    auto task = [scheduler]() -> coroutines::Task<> {
        co_await scheduler->yield_for(10ms);
    };

    coroutines::sync_wait(task());
}

TEST_F(TestCoroIoScheduler, YieldUntil)
{
    auto scheduler = coroutines::IoScheduler::get_instance();

    auto task = [scheduler]() -> coroutines::Task<> {
        co_await scheduler->yield_until(coroutines::clock_t::now() + 10ms);
    };

    coroutines::sync_wait(task());
}

TEST_F(TestCoroIoScheduler, Concurrent)
{
    auto scheduler = coroutines::IoScheduler::get_instance();

    auto task = [scheduler]() -> coroutines::Task<> {
        co_await scheduler->yield_for(10ms);
    };

    auto start = coroutines::clock_t::now();

    std::vector<coroutines::Task<>> tasks;

    for (uint32_t i = 0; i < 1000; i++)
    {
        tasks.push_back(task());
    }

    coroutines::sync_wait(coroutines::when_all(std::move(tasks)));

    ASSERT_LT(coroutines::clock_t::now() - start, 20ms);
}

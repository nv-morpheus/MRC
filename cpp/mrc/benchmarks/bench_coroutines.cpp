/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <benchmark/benchmark.h>

using namespace mrc;

static void mrc_coro_create_single_task_and_sync(benchmark::State& state)
{
    auto task = []() -> coroutines::Task<void> { co_return; };

    for (auto _ : state)
    {
        coroutines::sync_wait(task());
    }
}

static void mrc_coro_create_single_task_and_sync_on_when_all(benchmark::State& state)
{
    auto task = []() -> coroutines::Task<void> { co_return; };

    for (auto _ : state)
    {
        coroutines::sync_wait(coroutines::when_all(task()));
    }
}

static void mrc_coro_create_two_tasks_and_sync_on_when_all(benchmark::State& state)
{
    auto task = []() -> coroutines::Task<void> { co_return; };

    for (auto _ : state)
    {
        coroutines::sync_wait(coroutines::when_all(task(), task()));
    }
}

BENCHMARK(mrc_coro_create_single_task_and_sync);
BENCHMARK(mrc_coro_create_single_task_and_sync_on_when_all);
BENCHMARK(mrc_coro_create_two_tasks_and_sync_on_when_all);

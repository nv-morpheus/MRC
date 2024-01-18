/*
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

#include <benchmark/benchmark.h>
#include <boost/fiber/future/async.hpp>
#include <boost/fiber/future/future.hpp>
#include <boost/fiber/operations.hpp>
#include <boost/fiber/policy.hpp>

#include <memory>

static void boost_fibers_create_single_task_and_sync_post(benchmark::State& state)
{
    // warmup
    boost::fibers::async(boost::fibers::launch::post, [] {}).get();

    for (auto _ : state)
    {
        boost::fibers::async(boost::fibers::launch::post, [] {}).get();
    }
}

static void boost_fibers_create_single_task_and_sync_dispatch(benchmark::State& state)
{
    // warmup
    boost::fibers::async(boost::fibers::launch::dispatch, [] {}).get();

    for (auto _ : state)
    {
        boost::fibers::async(boost::fibers::launch::dispatch, [] {}).get();
    }
}

// static void boost_fibers_create_two_tasks_and_sync_on_when_all(benchmark::State& state)
// {
//     auto task = []() -> coroutines::Task<void> { co_return; };

//     for (auto _ : state)
//     {
//         coroutines::sync_wait(coroutines::when_all(task(), task()));
//     }
// }

static void boost_fibers_schedule(benchmark::State& state)
{
    // warmup
    boost::fibers::async(boost::fibers::launch::dispatch, [] {}).get();

    for (auto _ : state)
    {
        boost::this_fiber::yield();
    }
}

BENCHMARK(boost_fibers_create_single_task_and_sync_post);
BENCHMARK(boost_fibers_create_single_task_and_sync_dispatch);
// BENCHMARK(boost_fibers_create_two_tasks_and_sync_on_when_all);
BENCHMARK(boost_fibers_schedule);

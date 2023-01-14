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

#include "mrc/channel/v2/immediate_channel.hpp"
#include "mrc/coroutines/sync_wait.hpp"
#include "mrc/coroutines/task.hpp"
#include "mrc/coroutines/when_all.hpp"

#include <benchmark/benchmark.h>

#include <coroutine>

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

static void mrc_coro_await_suspend_never(benchmark::State& state)
{
    auto task = [&]() -> coroutines::Task<void> {
        for (auto _ : state)
        {
            co_await std::suspend_never{};
        }
        co_return;
    };

    coroutines::sync_wait(task());
}

// not-thread safe awaitable that returns a value
// this is an always ready non-yielding awaitable and should perform
// similar to a function call with the construction of the awaiter on the stack
class IncrementingAwaitable
{
    std::size_t m_counter{0};

    struct Awaiter
    {
        constexpr Awaiter(IncrementingAwaitable& parent) : m_parent(parent) {}

        constexpr static std::true_type await_ready() noexcept
        {
            return {};
        }

        constexpr static void await_suspend(std::coroutine_handle<> handle){};

        std::size_t await_resume() noexcept
        {
            return ++(m_parent.m_counter);
        }

        IncrementingAwaitable& m_parent;
    };

  public:
    [[nodiscard]] Awaiter operator co_await() noexcept
    {
        return {*this};
    }
};

static void mrc_coro_await_incrementing_awaitable(benchmark::State& state)
{
    IncrementingAwaitable awaitable;
    auto task = [&]() -> coroutines::Task<void> {
        std::size_t i;
        for (auto _ : state)
        {
            benchmark::DoNotOptimize(i = co_await awaitable);
        }
        co_return;
    };

    coroutines::sync_wait(task());
}

static void mrc_coro_await_incrementing_awaitable_baseline(benchmark::State& state)
{
    auto task = [&]() -> coroutines::Task<void> {
        std::size_t i{0};
        std::size_t j{0};
        for (auto _ : state)
        {
            benchmark::DoNotOptimize(i = ++j);
        }
        co_return;
    };

    coroutines::sync_wait(task());
}

static void mrc_coro_immediate_channel(benchmark::State& state)
{
    channel::v2::ImmediateChannel<std::size_t> immediate_channel;

    auto src = [&]() -> coroutines::Task<> {
        for (auto _ : state)
        {
            co_await immediate_channel.async_write(42);
        }
        immediate_channel.close();
        co_return;
    };

    auto sink = [&]() -> coroutines::Task<> {
        while (auto val = co_await immediate_channel.async_read()) {}
        co_return;
    };

    coroutines::sync_wait(coroutines::when_all(sink(), src()));
}

static auto bar(std::size_t i) -> std::size_t
{
    return i += 5;
}

static void foo(std::size_t i)
{
    benchmark::DoNotOptimize(bar(i));
}

static void mrc_coro_immedate_channel_composite_fn_baseline(benchmark::State& state)
{
    auto task = [&]() -> coroutines::Task<> {
        for (auto _ : state)
        {
            foo(42);
        }
        co_return;
    };

    coroutines::sync_wait(task());
}

BENCHMARK(mrc_coro_create_single_task_and_sync);
BENCHMARK(mrc_coro_create_single_task_and_sync_on_when_all);
BENCHMARK(mrc_coro_create_two_tasks_and_sync_on_when_all);
BENCHMARK(mrc_coro_await_suspend_never);
BENCHMARK(mrc_coro_await_incrementing_awaitable_baseline);
BENCHMARK(mrc_coro_await_incrementing_awaitable);
BENCHMARK(mrc_coro_immediate_channel);
BENCHMARK(mrc_coro_immedate_channel_composite_fn_baseline);

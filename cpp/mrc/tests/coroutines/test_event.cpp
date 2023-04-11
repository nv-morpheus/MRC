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

#include "mrc/coroutines/concepts/awaitable.hpp"
#include "mrc/coroutines/event.hpp"
#include "mrc/coroutines/sync_wait.hpp"
#include "mrc/coroutines/task.hpp"
#include "mrc/coroutines/thread_pool.hpp"
#include "mrc/coroutines/when_all.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <coroutine>
#include <cstdint>
#include <tuple>

using namespace mrc;

class TestCoroEvent : public ::testing::Test
{};

TEST_F(TestCoroEvent, LifeCycle)
{
    coroutines::Event e{};

    auto func = [&]() -> coroutines::Task<uint64_t> {
        co_await e;
        co_return 42;
    };

    auto task = func();

    task.resume();
    EXPECT_FALSE(task.is_ready());
    e.set();  // this will automaticaly resume the task that is awaiting the event.
    EXPECT_TRUE(task.is_ready());
    EXPECT_TRUE(task.promise().result() == 42);
}

auto producer(coroutines::Event& event) -> void
{
    // Long running task that consumers are waiting for goes here...
    event.set();
}

auto consumer(const coroutines::Event& event) -> coroutines::Task<uint64_t>
{
    co_await event;
    // Normally consume from some object which has the stored result from the producer
    co_return 42;
}

// TEST_CASE("event one watcher", "[event]")
TEST_F(TestCoroEvent, SingleWatcher)
{
    coroutines::Event e{};

    auto value = consumer(e);
    value.resume();  // start co_awaiting event
    EXPECT_FALSE(value.is_ready());

    producer(e);

    EXPECT_TRUE(value.promise().result() == 42);
}

// TEST_CASE("event multiple watchers", "[event]")
TEST_F(TestCoroEvent, MultipleWatchers)
{
    coroutines::Event e{};

    auto value1 = consumer(e);
    auto value2 = consumer(e);
    auto value3 = consumer(e);
    value1.resume();  // start co_awaiting event
    value2.resume();
    value3.resume();
    EXPECT_FALSE(value1.is_ready());
    EXPECT_FALSE(value2.is_ready());
    EXPECT_FALSE(value3.is_ready());

    producer(e);

    EXPECT_TRUE(value1.promise().result() == 42);
    EXPECT_TRUE(value2.promise().result() == 42);
    EXPECT_TRUE(value3.promise().result() == 42);
}

// TEST_CASE("event reset", "[event]")
TEST_F(TestCoroEvent, Reset)
{
    coroutines::Event e{};

    e.reset();
    EXPECT_FALSE(e.is_set());

    auto value1 = consumer(e);
    value1.resume();  // start co_awaiting event
    EXPECT_FALSE(value1.is_ready());

    producer(e);
    EXPECT_TRUE(value1.promise().result() == 42);

    e.reset();

    auto value2 = consumer(e);
    value2.resume();
    EXPECT_FALSE(value2.is_ready());

    producer(e);

    EXPECT_TRUE(value2.promise().result() == 42);
}

// TEST_CASE("event fifo", "[event]")
TEST_F(TestCoroEvent, FIFO)
{
    coroutines::Event e{};

    // Need consistency FIFO on a single thread to verify the execution order is correct.
    coroutines::ThreadPool tp{coroutines::ThreadPool::Options{.thread_count = 1}};

    std::atomic<uint64_t> counter{0};

    auto make_waiter = [&](uint64_t value) -> coroutines::Task<void> {
        co_await tp.schedule();
        co_await e;

        counter++;
        EXPECT_TRUE(counter == value);

        co_return;
    };

    auto make_setter = [&]() -> coroutines::Task<void> {
        co_await tp.schedule();
        EXPECT_TRUE(counter == 0);
        e.set(coroutines::ResumeOrderPolicy::fifo);
        co_return;
    };

    coroutines::sync_wait(coroutines::when_all(make_waiter(1),
                                               make_waiter(2),
                                               make_waiter(3),
                                               make_waiter(4),
                                               make_waiter(5),
                                               make_setter()));

    EXPECT_TRUE(counter == 5);
}

// TEST_CASE("event fifo none", "[event]")
TEST_F(TestCoroEvent, FIFO_None)
{
    coroutines::Event e{};

    // Need consistency FIFO on a single thread to verify the execution order is correct.
    coroutines::ThreadPool tp{coroutines::ThreadPool::Options{.thread_count = 1}};

    std::atomic<uint64_t> counter{0};

    auto make_setter = [&]() -> coroutines::Task<void> {
        co_await tp.schedule();
        EXPECT_TRUE(counter == 0);
        e.set(coroutines::ResumeOrderPolicy::fifo);
        co_return;
    };

    coroutines::sync_wait(coroutines::when_all(make_setter()));

    EXPECT_TRUE(counter == 0);
}

// TEST_CASE("event fifo single", "[event]")
TEST_F(TestCoroEvent, FIFO_Single)
{
    coroutines::Event e{};

    // Need consistency FIFO on a single thread to verify the execution order is correct.
    coroutines::ThreadPool tp{coroutines::ThreadPool::Options{.thread_count = 1}};

    std::atomic<uint64_t> counter{0};

    auto make_waiter = [&](uint64_t value) -> coroutines::Task<void> {
        co_await tp.schedule();
        co_await e;

        counter++;
        EXPECT_TRUE(counter == value);

        co_return;
    };

    auto make_setter = [&]() -> coroutines::Task<void> {
        co_await tp.schedule();
        EXPECT_TRUE(counter == 0);
        e.set(coroutines::ResumeOrderPolicy::fifo);
        co_return;
    };

    coroutines::sync_wait(coroutines::when_all(make_waiter(1), make_setter()));

    EXPECT_TRUE(counter == 1);
}

// TEST_CASE("event fifo executor", "[event]")
TEST_F(TestCoroEvent, FIFO_Executor)
{
    coroutines::Event e{};

    // Need consistency FIFO on a single thread to verify the execution order is correct.
    coroutines::ThreadPool tp{coroutines::ThreadPool::Options{.thread_count = 1}};

    std::atomic<uint64_t> counter{0};

    auto make_waiter = [&](uint64_t value) -> coroutines::Task<void> {
        co_await tp.schedule();
        co_await e;

        counter++;
        EXPECT_TRUE(counter == value);

        co_return;
    };

    auto make_setter = [&]() -> coroutines::Task<void> {
        co_await tp.schedule();
        EXPECT_TRUE(counter == 0);
        e.set(tp, coroutines::ResumeOrderPolicy::fifo);
        co_return;
    };

    coroutines::sync_wait(coroutines::when_all(make_waiter(1),
                                               make_waiter(2),
                                               make_waiter(3),
                                               make_waiter(4),
                                               make_waiter(5),
                                               make_setter()));

    EXPECT_TRUE(counter == 5);
}

// TEST_CASE("event fifo none executor", "[event]")
TEST_F(TestCoroEvent, FIFO_NoExecutor)
{
    coroutines::Event e{};

    // Need consistency FIFO on a single thread to verify the execution order is correct.
    coroutines::ThreadPool tp{coroutines::ThreadPool::Options{.thread_count = 1}};

    std::atomic<uint64_t> counter{0};

    auto make_setter = [&]() -> coroutines::Task<void> {
        co_await tp.schedule();
        EXPECT_TRUE(counter == 0);
        e.set(tp, coroutines::ResumeOrderPolicy::fifo);
        co_return;
    };

    coroutines::sync_wait(coroutines::when_all(make_setter()));

    EXPECT_TRUE(counter == 0);
}

// TEST_CASE("event fifo single executor", "[event]")
TEST_F(TestCoroEvent, FIFO_SingleExecutor)
{
    coroutines::Event e{};

    // Need consistency FIFO on a single thread to verify the execution order is correct.
    coroutines::ThreadPool tp{coroutines::ThreadPool::Options{.thread_count = 1}};

    std::atomic<uint64_t> counter{0};

    auto make_waiter = [&](uint64_t value) -> coroutines::Task<void> {
        co_await tp.schedule();
        co_await e;

        counter++;
        EXPECT_TRUE(counter == value);

        co_return;
    };

    auto make_setter = [&]() -> coroutines::Task<void> {
        co_await tp.schedule();
        EXPECT_TRUE(counter == 0);
        e.set(tp, coroutines::ResumeOrderPolicy::fifo);
        co_return;
    };

    coroutines::sync_wait(coroutines::when_all(make_waiter(1), make_setter()));

    EXPECT_TRUE(counter == 1);
}

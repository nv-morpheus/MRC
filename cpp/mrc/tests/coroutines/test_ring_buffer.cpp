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
#include "mrc/coroutines/latch.hpp"
#include "mrc/coroutines/ring_buffer.hpp"
#include "mrc/coroutines/sync_wait.hpp"
#include "mrc/coroutines/task.hpp"
#include "mrc/coroutines/thread_pool.hpp"
#include "mrc/coroutines/when_all.hpp"

#include <gtest/gtest.h>

#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

using namespace mrc;

class TestCoroRingBuffer : public ::testing::Test
{};

// TEST_CASE("ring_buffer zero num_elements", "[ring_buffer]")
TEST_F(TestCoroRingBuffer, ZeroCapacity)
{
    EXPECT_ANY_THROW(coroutines::RingBuffer<uint64_t> rb{{.capacity = 0}});
}

// TEST_CASE("ring_buffer single element", "[ring_buffer]")
TEST_F(TestCoroRingBuffer, SingleElement)
{
    const size_t iterations = 10;
    coroutines::RingBuffer<uint64_t> rb{{.capacity = 1}};

    std::vector<uint64_t> output{};

    auto make_producer_task = [&]() -> coroutines::Task<void> {
        for (size_t i = 1; i <= iterations; ++i)
        {
            co_await rb.write(i);
        }
        co_return;
    };

    auto make_consumer_task = [&]() -> coroutines::Task<void> {
        for (size_t i = 1; i <= iterations; ++i)
        {
            auto expected = co_await rb.read();
            auto value    = std::move(*expected);

            output.emplace_back(std::move(value));
        }
        co_return;
    };

    coroutines::sync_wait(coroutines::when_all(make_producer_task(), make_consumer_task()));

    for (size_t i = 1; i <= iterations; ++i)
    {
        EXPECT_TRUE(output[i - 1] == i);
    }

    EXPECT_TRUE(rb.empty());
}

TEST_F(TestCoroRingBuffer, WriteX5ThenClose)
{
    const size_t iterations = 5;
    coroutines::RingBuffer<uint64_t> rb{{.capacity = 2}};

    std::vector<uint64_t> output{};

    auto make_producer_task = [&]() -> coroutines::Task<void> {
        for (size_t i = 1; i <= iterations; ++i)
        {
            EXPECT_FALSE(rb.is_closed());
            co_await rb.write(i);
        }
        rb.close();
        EXPECT_TRUE(rb.is_closed());
        auto status = co_await rb.write(42);
        EXPECT_EQ(status, coroutines::RingBufferOpStatus::Stopped);
        co_return;
    };

    auto make_consumer_task = [&]() -> coroutines::Task<void> {
        while (true)
        {
            auto expected = co_await rb.read();

            if (!expected)
            {
                break;
            }
            auto value = std::move(*expected);
            output.emplace_back(std::move(value));
        }
        co_return;
    };

    coroutines::sync_wait(coroutines::when_all(make_producer_task(), make_consumer_task()));

    for (size_t i = 1; i <= iterations; ++i)
    {
        EXPECT_TRUE(output[i - 1] == i);
    }

    EXPECT_TRUE(rb.empty());
    EXPECT_TRUE(rb.is_closed());
}

TEST_F(TestCoroRingBuffer, FullyBufferedWriteX5ThenClose)
{
    const size_t iterations = 5;
    coroutines::RingBuffer<uint64_t> rb{{.capacity = 16}};
    coroutines::Latch latch{iterations + 1};

    std::vector<uint64_t> output{};

    auto make_producer_task = [&]() -> coroutines::Task<void> {
        for (size_t i = 1; i <= iterations; ++i)
        {
            EXPECT_FALSE(rb.is_closed());
            co_await rb.write(i);
            latch.count_down();
        }
        rb.close();
        EXPECT_TRUE(rb.is_closed());
        auto status = co_await rb.write(42);
        EXPECT_EQ(status, coroutines::RingBufferOpStatus::Stopped);
        latch.count_down();
        co_return;
    };

    auto make_consumer_task = [&]() -> coroutines::Task<void> {
        co_await latch;
        while (true)
        {
            auto expected = co_await rb.read();

            if (!expected)
            {
                break;
            }
            auto value = std::move(*expected);
            output.emplace_back(std::move(value));
        }
        co_return;
    };

    coroutines::sync_wait(coroutines::when_all(make_producer_task(), make_consumer_task()));

    for (size_t i = 1; i <= iterations; ++i)
    {
        EXPECT_TRUE(output[i - 1] == i);
    }

    EXPECT_TRUE(rb.empty());
    EXPECT_TRUE(rb.is_closed());
}

// TEST_CASE("ring_buffer many elements many producers many consumers", "[ring_buffer]")
TEST_F(TestCoroRingBuffer, MultiProducerMultiConsumer)
{
    const size_t iterations = 1'000'000;
    const size_t consumers  = 100;
    const size_t producers  = 100;

    coroutines::ThreadPool tp{{.thread_count = 4}};
    coroutines::RingBuffer<uint64_t> rb{{.capacity = 64}};
    coroutines::Latch producers_latch{producers};

    auto make_producer_task = [&]() -> coroutines::Task<void> {
        co_await tp.schedule();
        auto to_produce = iterations / producers;

        for (size_t i = 1; i <= to_produce; ++i)
        {
            switch (i % 3)
            {
            case 0:
                co_await rb.write(i);
                break;
            case 1:
                co_await rb.write(i).resume_immediately();
                break;
            case 2:
                co_await rb.write(i).resume_on(&tp);
                break;
            }
        }

        producers_latch.count_down();

        co_return;
    };

    auto make_consumer_task = [&]() -> coroutines::Task<void> {
        co_await tp.schedule();

        while (true)
        {
            auto expected = co_await rb.read();
            if (!expected)
            {
                break;
            }

            auto item = std::move(*expected);

            co_await tp.yield();  // mimic some work
        }

        co_return;
    };

    auto make_shutdown_task = [&]() -> coroutines::Task<void> {
        co_await tp.schedule();
        co_await producers_latch;
        rb.close();
        co_return;
    };

    std::vector<coroutines::Task<void>> tasks{};
    tasks.reserve(consumers + producers + 1);

    tasks.emplace_back(make_shutdown_task());

    for (size_t i = 0; i < consumers; ++i)
    {
        tasks.emplace_back(make_consumer_task());
    }
    for (size_t i = 0; i < producers; ++i)
    {
        tasks.emplace_back(make_producer_task());
    }

    EXPECT_EQ(tasks.size(), consumers + producers + 1);

    coroutines::sync_wait(coroutines::when_all(std::move(tasks)));

    EXPECT_TRUE(rb.empty());
}

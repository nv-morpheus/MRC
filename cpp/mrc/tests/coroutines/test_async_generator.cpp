/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "mrc/coroutines/sync_wait.hpp"
#include "mrc/coroutines/task.hpp"

#include <gtest/gtest.h>

#include <coroutine>

using namespace mrc;

class TestCoroAsyncGenerator : public ::testing::Test
{};

TEST_F(TestCoroAsyncGenerator, Iterator)
{
    auto generator = []() -> coroutines::AsyncGenerator<int> {
        for (int i = 0; i < 2; i++)
        {
            co_yield i;
        }
    }();

    auto task = [&]() -> coroutines::Task<> {
        auto iter = co_await generator.begin();

        EXPECT_TRUE(iter);
        EXPECT_EQ(*iter, 0);
        EXPECT_NE(iter, generator.end());

        co_await ++iter;

        EXPECT_TRUE(iter);
        EXPECT_EQ(*iter, 1);
        EXPECT_NE(iter, generator.end());

        co_await ++iter;
        EXPECT_FALSE(iter);
        EXPECT_EQ(iter, generator.end());

        co_return;
    };

    coroutines::sync_wait(task());
}

TEST_F(TestCoroAsyncGenerator, LoopOnGenerator)
{
    auto generator = []() -> coroutines::AsyncGenerator<int> {
        for (int i = 0; i < 2; i++)
        {
            co_yield i;
        }
    }();

    auto task = [&]() -> coroutines::Task<> {
        for (int i = 0; i < 2; i++)
        {
            auto iter = co_await generator.begin();

            EXPECT_TRUE(iter);
            EXPECT_EQ(*iter, 0);
            EXPECT_NE(iter, generator.end());

            co_await ++iter;

            EXPECT_TRUE(iter);
            EXPECT_EQ(*iter, 1);
            EXPECT_NE(iter, generator.end());

            co_await ++iter;
            EXPECT_FALSE(iter);
            EXPECT_EQ(iter, generator.end());

            co_return;
        }
    };

    coroutines::sync_wait(task());
}

TEST_F(TestCoroAsyncGenerator, MultipleBegins)
{
    auto generator = []() -> coroutines::AsyncGenerator<int> {
        for (int i = 0; i < 2; i++)
        {
            co_yield i;
        }
    }();

    // this test shows that begin() and operator++() perform essentially the same function
    // both advance the generator to the next state
    // while a generator is an iterable, it doesn't hold the entire sequence in memory, it does
    // what it suggests, it generates the next item from the previous

    auto task = [&]() -> coroutines::Task<> {
        auto iter = co_await generator.begin();

        EXPECT_TRUE(iter);
        EXPECT_EQ(*iter, 0);
        EXPECT_NE(iter, generator.end());

        iter = co_await generator.begin();

        EXPECT_TRUE(iter);
        EXPECT_EQ(*iter, 1);
        EXPECT_NE(iter, generator.end());

        iter = co_await generator.begin();
        EXPECT_FALSE(iter);
        EXPECT_EQ(iter, generator.end());

        co_return;
    };

    coroutines::sync_wait(task());
}

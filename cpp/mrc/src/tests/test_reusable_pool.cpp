/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/channel/status.hpp"
#include "mrc/data/reusable_pool.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>

using namespace mrc;

// iwyu is getting confused between std::uint32_t and boost::uint32_t
// IWYU pragma: no_include <boost/cstdint.hpp>

class TestReusablePool : public ::testing::Test
{};

TEST_F(TestReusablePool, LifeCycle)
{
    auto pool = data::ReusablePool<int>::create(4);
}

TEST_F(TestReusablePool, Capacity)
{
    auto pool = data::ReusablePool<int>::create(4);

    pool->emplace(1);
    pool->emplace(2);
    pool->emplace(3);

    // the number of items manages by the pool must be less (not equal to) the capacity
    EXPECT_ANY_THROW(pool->emplace(4));
}

TEST_F(TestReusablePool, Reset)
{
    std::atomic<std::size_t> counter = 0;

    auto pool = data::ReusablePool<int>::create(4, [&](int& i) {
        i = 42;
        counter++;
    });

    pool->emplace(0);
    pool->emplace(1);
    pool->emplace(2);

    for (int i = 0; i < 3; i++)
    {
        auto reusable_int = pool->await_item();
        EXPECT_EQ(*reusable_int, i);
    }

    // all our initial values should now be reset to 42
    for (int i = 0; i < 10; i++)
    {
        auto reusable_int = pool->await_item();
        EXPECT_EQ(*reusable_int, 42);
    }

    EXPECT_EQ(counter, 13);
}

TEST_F(TestReusablePool, Immutable)
{
    std::atomic<std::size_t> counter = 0;

    auto pool = data::ReusablePool<int>::create(4, [&](int& i) {
        i = 42;
        counter++;
    });

    pool->emplace(0);
    pool->emplace(1);
    pool->emplace(2);

    for (int i = 0; i < 3; i++)
    {
        auto reusable_int = pool->await_item();
        EXPECT_EQ(*reusable_int, i);
    }

    // hold the last iteration as a shared_immutable
    data::SharedReusable<int> shared_immutable;

    // all our initial values should now be reset to 42
    for (int i = 0; i < 10; i++)
    {
        auto reusable_int = pool->await_item();
        EXPECT_EQ(*reusable_int, 42);
        shared_immutable = std::move(reusable_int);
    }

    EXPECT_EQ(counter, 12);

    // *shared_immutable = 42; // no write access

    // release the last one
    shared_immutable.release();

    EXPECT_EQ(counter, 13);
}

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

#include "internal/utils/ranges.hpp"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

using namespace mrc;

class TestRanges : public ::testing::Test
{};

TEST_F(TestRanges, FindRanges0)
{
    std::vector<int> a{1};
    std::vector<std::pair<int, int>> a_ranges{{1, 1}};
    auto ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1");
}

TEST_F(TestRanges, FindRanges1)
{
    std::vector<int> a{1, 2};
    std::vector<std::pair<int, int>> a_ranges{{1, 2}};
    auto ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1-2");
}

TEST_F(TestRanges, FindRanges2)
{
    std::vector<int> a{1, 2, 3};
    std::vector<std::pair<int, int>> a_ranges{{1, 3}};
    auto ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1-3");
}

TEST_F(TestRanges, FindRanges3)
{
    std::vector<int> a{1, 3};
    std::vector<std::pair<int, int>> a_ranges{{1, 1}, {3, 3}};
    auto ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1,3");
}

TEST_F(TestRanges, FindRanges4)
{
    std::vector<int> a{1, 2, 4, 5, 6, 10};
    std::vector<std::pair<int, int>> a_ranges{{1, 2}, {4, 6}, {10, 10}};
    auto ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "1-2,4-6,10");
}

TEST_F(TestRanges, FindRanges5)
{
    std::vector<int> a{0, 1, 2, 3, 4, 5, 6};
    std::vector<std::pair<int, int>> a_ranges{{0, 6}};
    auto ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "0-6");
}

TEST_F(TestRanges, FindRanges6)
{
    std::vector<long> a{0, 1, 2, 3, 5, 6};
    std::vector<std::pair<long, long>> a_ranges{{0, 3}, {5, 6}};
    auto ranges = find_ranges(a);
    ASSERT_EQ(ranges, a_ranges);
    ASSERT_EQ(print_ranges(ranges), "0-3,5-6");
}

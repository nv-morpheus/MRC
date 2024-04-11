/*
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

#include "./test_mrc.hpp"  // IWYU pragma: associated

#include "mrc/utils/string_utils.hpp"  // for split_string_to_vector

#include <gtest/gtest.h>  // for EXPECT_EQ

#include <string>
#include <vector>

namespace mrc {

TEST_CLASS(StringUtils);

TEST_F(TestStringUtils, TestSplitStringToVector)
{
    struct TestValues
    {
        std::string str;
        std::string delimiter;
        std::vector<std::string> expected_result;
    };

    std::vector<TestValues> values = {
        {"Hello,World,!", ",", {"Hello", "World", "!"}},
        {"a/b/c", "/", {"a", "b", "c"}},
        {"/a/b/c", "/", {"", "a", "b", "c"}},  // leading delimeter
        {"a/b/c/", "/", {"a", "b", "c", ""}},  // trailing delimeter
        {"abcd", "/", {"abcd"}},               // no delimeter
        {"", "/", {""}},                       // empty string
        {"/", "/", {"", ""}},                  // single delimeter
        {"//", "/", {"", "", ""}},             // duplicate delimeter
    };

    for (const auto& value : values)
    {
        auto result = mrc::split_string_to_vector(value.str, value.delimiter);
        EXPECT_EQ(result, value.expected_result);
    }
}

}  // namespace mrc

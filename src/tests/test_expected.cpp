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

#include "internal/expected.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <memory>
#include <ostream>
#include <string>
#include <utility>

using namespace mrc;
using namespace mrc::internal;

class TestExpected : public ::testing::Test
{};

static Expected<> make_void()
{
    return {};
}

static Expected<> make_void_fail()
{
    return Error::create("void fail");
}

static Expected<int> make_int(int i)
{
    return i;
}

static Expected<int> make_int_fail(int i = 0)
{
    return Error::create("int fail");
}

static Expected<std::string> make_str(std::string str)
{
    return str;
}

static Expected<std::string> make_str_fail(std::string str = "")
{
    return Error::create("str fail");
}

TEST_F(TestExpected, Chaining0)
{
    auto rc = make_int(42).transform([&](auto& i) {
        std::stringstream ss;
        ss << i;
        return make_str(ss.str());
    });

    EXPECT_TRUE(rc);
    EXPECT_TRUE(rc->value() == "42");
}

TEST_F(TestExpected, Chaining1)
{
    auto rc = make_int_fail(42).transform([&](auto& i) {
        std::stringstream ss;
        ss << i;
        return make_str(ss.str());
    });

    EXPECT_FALSE(rc);
    EXPECT_EQ(rc.error().code(), ErrorCode::Internal);
    EXPECT_EQ(rc.error().message(), "int fail");
}

TEST_F(TestExpected, Chaining2)
{
    auto rc = make_void().transform([&] {
        std::stringstream ss;
        ss << 42;
        return make_str(ss.str());
    });

    EXPECT_TRUE(rc);
    EXPECT_TRUE(*rc == "42");
}

TEST_F(TestExpected, Chaining3)
{
    auto rc = make_void_fail().transform([&] {
        std::stringstream ss;
        ss << 42;
        return make_str(ss.str());
    });

    EXPECT_FALSE(rc);
    EXPECT_EQ(rc.error().message(), "void fail");
    EXPECT_ANY_THROW(rc->value());
}

TEST_F(TestExpected, UniquePointer)
{
    Expected<std::unique_ptr<int>> rc = std::make_unique<int>(42);

    EXPECT_TRUE(rc);
    EXPECT_TRUE(rc.value());
    EXPECT_EQ(*rc.value(), 42);

    auto ptr = std::move(rc.value());

    EXPECT_EQ(*ptr, 42);
    EXPECT_TRUE(rc);
    EXPECT_FALSE(rc.value());
}

TEST_F(TestExpected, Examples)
{
    EXPECT_TRUE(make_void().and_then(make_void));
    EXPECT_EQ(make_void().transform([] { return make_int(1); }).value(), 1);

    auto mul2 = [](int a) { return a * 2; };
    auto inc1 = [](int a) { return a + 1; };

    {
        Expected<int> e = 21;
        auto ret        = e.map(mul2);
        EXPECT_TRUE(ret);
        EXPECT_EQ(ret.value(), 42);
    }

    {
        auto ret = make_int(21).map(mul2).map(inc1);
        EXPECT_TRUE(ret);
        EXPECT_EQ(ret.value(), 43);
    }

    {
        Expected<int> e = 21;
        auto ret        = e.map(mul2).map(mul2);
        EXPECT_TRUE(ret);
        EXPECT_EQ(ret.value(), 84);
    }
}

TEST_F(TestExpected, OrElse)
{
    auto status = make_int_fail(42).or_else([](auto& e) { LOG(INFO) << e.message(); });
    EXPECT_FALSE(status);
}

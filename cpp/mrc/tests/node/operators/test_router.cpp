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

#include "../../test_mrc.hpp"  // IWYU pragma: associated
#include "../test_nodes.hpp"

#include "mrc/exceptions/runtime_error.hpp"  // for MrcRuntimeError
#include "mrc/node/operators/router.hpp"

#include <gtest/gtest.h>

TEST_CLASS(Router);
namespace {
template <typename T>
std::string even_odd(const T& t)
{
    return t % 2 == 1 ? "odd" : "even";
}
};  // namespace

namespace mrc::node {

template <typename T>
class TestStaticRouterComponent : public StaticRouterComponentBase<std::string, T>
{
  public:
    using base_t = StaticRouterComponentBase<std::string, T>;

    TestStaticRouterComponent() : base_t(std::vector<std::string>{"odd", "even"}) {}

  protected:
    std::string determine_key_for_value(const T& t) override
    {
        return even_odd(t);
    }
};

}  // namespace mrc::node

namespace mrc {

TEST_F(TestRouter, StaticRouterComponent_SourceToRouterToSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto router = std::make_shared<node::TestStaticRouterComponent<int>>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source("odd"), *sink1);
    mrc::make_edge(*router->get_source("even"), *sink2);

    source->run();
    sink1->run();
    sink2->run();

    EXPECT_EQ((std::vector<int>{1}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{0, 2}), sink2->get_values());
}

TEST_F(TestRouter, StaticRouterComponent_SourceToRouterToDifferentSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto router = std::make_shared<node::TestStaticRouterComponent<int>>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source("odd"), *sink1);
    mrc::make_edge(*router->get_source("even"), *sink2);

    source->run();
    sink1->run();

    EXPECT_EQ((std::vector<int>{1}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{0, 2}), sink2->get_values());
}

TEST_F(TestRouter, LambdaStaticRouterComponent_SourceToRouterToSinks)
{
    auto source = std::make_shared<node::TestSource<int>>(10);
    auto router = std::make_shared<node::LambdaStaticRouterComponent<int, int>>(std::vector<int>{1, 2, 3},
                                                                                [](const int& data) {
                                                                                    return (data % 3) + 1;
                                                                                });
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();
    auto sink3  = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source(1), *sink1);
    mrc::make_edge(*router->get_source(2), *sink2);
    mrc::make_edge(*router->get_source(3), *sink3);

    source->run();
    sink1->run();
    sink2->run();
    sink3->run();

    EXPECT_EQ((std::vector<int>{0, 3, 6, 9}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{1, 4, 7}), sink2->get_values());
    EXPECT_EQ((std::vector<int>{2, 5, 8}), sink3->get_values());
}

TEST_F(TestRouter, LambdaStaticRouterComponent_SourceToRouterToDifferentSinks)
{
    auto source = std::make_shared<node::TestSource<int>>(10);
    auto router = std::make_shared<node::LambdaStaticRouterComponent<int, int>>(std::vector<int>{1, 2, 3},
                                                                                [](const int& data) {
                                                                                    return (data % 3) + 1;
                                                                                });
    auto sink1  = std::make_shared<node::TestSinkComponent<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();
    auto sink3  = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source(1), *sink1);
    mrc::make_edge(*router->get_source(2), *sink2);
    mrc::make_edge(*router->get_source(3), *sink3);

    source->run();
    sink2->run();

    EXPECT_EQ((std::vector<int>{0, 3, 6, 9}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{1, 4, 7}), sink2->get_values());
    EXPECT_EQ((std::vector<int>{2, 5, 8}), sink3->get_values());
}

TEST_F(TestRouter, LambdaRouter_SourceToRouterToSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto router = std::make_shared<node::LambdaRouter<std::string, int>>(&even_odd<int>);
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source("odd"), *sink1);
    mrc::make_edge(*router->get_source("even"), *sink2);

    source->run();
    sink1->run();
    sink2->run();

    EXPECT_EQ((std::vector<int>{1}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{0, 2}), sink2->get_values());
}

TEST_F(TestRouter, LambdaRouter_SourceToRouterToDifferentSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto router = std::make_shared<node::LambdaRouter<std::string, int>>(&even_odd<int>);
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source("odd"), *sink1);
    mrc::make_edge(*router->get_source("even"), *sink2);

    source->run();
    sink1->run();

    EXPECT_EQ((std::vector<int>{1}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{0, 2}), sink2->get_values());
}

// two possible errors, either the key function throws an error, or the key function returns a key that is invalid
TEST_F(TestRouter, LambdaRouterOnKeyError)
{
    auto source = std::make_shared<node::TestSource<int>>(3);
    auto router = std::make_shared<node::LambdaRouter<int, int>>([](const int& data) {
        if (data == 2)
        {
            throw std::runtime_error("Test Error");
        }

        return data + 1;
    });

    auto sink1 = std::make_shared<node::TestSink<int>>();
    auto sink2 = std::make_shared<node::TestSink<int>>();
    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source(1), *sink1);
    mrc::make_edge(*router->get_source(2), *sink2);

    EXPECT_THROW(
        {
            source->run();
            sink2->run();
            sink2->run();
        },
        exceptions::MrcRuntimeError);
}

TEST_F(TestRouter, LambdaRouterOnKeyInvalidValue)
{
    auto source = std::make_shared<node::TestSource<int>>(3);
    auto router = std::make_shared<node::LambdaRouter<int, int>>([](const int& data) {
        return data + 1;  // On the third value, this will return 4, which is not a valid key
    });

    auto sink1 = std::make_shared<node::TestSink<int>>();
    auto sink2 = std::make_shared<node::TestSink<int>>();
    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source(1), *sink1);
    mrc::make_edge(*router->get_source(2), *sink2);

    EXPECT_THROW(
        {
            source->run();
            sink2->run();
            sink2->run();
        },
        exceptions::MrcRuntimeError);
}

}  // namespace mrc

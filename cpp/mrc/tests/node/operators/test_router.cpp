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

#include "mrc/node/operators/router.hpp"

#include <gtest/gtest.h>

TEST_CLASS(Router);

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
        return t % 2 == 1 ? "odd" : "even";
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

}  // namespace mrc

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

#include "mrc/edge/edge_builder.hpp"
#include "mrc/edge/edge_writable.hpp"
#include "mrc/node/operators/combine_latest.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace mrc {

TEST_CLASS(CombineLatest);

TEST_F(TestCombineLatest, SourceToCLToSink)
{
    auto source1 = std::make_shared<node::TestSource<int>>();
    auto source2 = std::make_shared<node::TestSource<float>>();

    auto combine_latest = std::make_shared<node::CombineLatestComponent<std::tuple<int, float>>>();

    auto sink = std::make_shared<node::TestSink<std::tuple<int, float>>>();

    mrc::make_edge(*source1, *combine_latest->get_sink<0>());
    mrc::make_edge(*source2, *combine_latest->get_sink<1>());
    mrc::make_edge(*combine_latest, *sink);

    source1->run();
    source2->run();

    sink->run();

    EXPECT_EQ(sink->get_values(),
              (std::vector<std::tuple<int, float>>{
                  std::tuple<int, float>{2, 0},
                  std::tuple<int, float>{2, 1},
                  std::tuple<int, float>{2, 2},
              }));
}

TEST_F(TestCombineLatest, SourceToCLToSinkComponent)
{
    auto source1 = std::make_shared<node::TestSource<int>>();
    auto source2 = std::make_shared<node::TestSource<float>>();

    auto combine_latest = std::make_shared<node::CombineLatestComponent<std::tuple<int, float>>>();

    auto sink = std::make_shared<node::TestSinkComponent<std::tuple<int, float>>>();

    mrc::make_edge(*source1, *combine_latest->get_sink<0>());
    mrc::make_edge(*source2, *combine_latest->get_sink<1>());
    mrc::make_edge(*combine_latest, *sink);

    source1->run();
    source2->run();

    EXPECT_EQ(sink->get_values(),
              (std::vector<std::tuple<int, float>>{
                  std::tuple<int, float>{2, 0},
                  std::tuple<int, float>{2, 1},
                  std::tuple<int, float>{2, 2},
              }));
}

TEST_F(TestCombineLatest, PrimaryFirst)
{
    auto source1 = std::make_shared<node::TestSource<int>>();
    auto source2 = std::make_shared<node::TestSource<float>>();

    auto combine_latest = std::make_shared<node::CombineLatestComponent<std::tuple<int, float>>>();

    auto sink = std::make_shared<node::TestSink<std::tuple<int, float>>>();

    mrc::make_edge(*source1, *combine_latest->get_sink<0>());
    mrc::make_edge(*source2, *combine_latest->get_sink<1>());
    mrc::make_edge(*combine_latest, *sink);

    source1->run();
    source2->run();

    sink->run();

    EXPECT_EQ(sink->get_values(),
              (std::vector<std::tuple<int, float>>{
                  std::tuple<int, float>{2, 0},
                  std::tuple<int, float>{2, 1},
                  std::tuple<int, float>{2, 2},
              }));
}

TEST_F(TestCombineLatest, SecondaryFirst)
{
    auto source1 = std::make_shared<node::TestSource<int>>();
    auto source2 = std::make_shared<node::TestSource<float>>();

    auto combine_latest = std::make_shared<node::CombineLatestComponent<std::tuple<int, float>>>();

    auto sink = std::make_shared<node::TestSink<std::tuple<int, float>>>();

    mrc::make_edge(*source1, *combine_latest->get_sink<0>());
    mrc::make_edge(*source2, *combine_latest->get_sink<1>());
    mrc::make_edge(*combine_latest, *sink);

    source2->run();
    source1->run();

    sink->run();

    EXPECT_EQ(sink->get_values(),
              (std::vector<std::tuple<int, float>>{
                  std::tuple<int, float>{0, 2},
                  std::tuple<int, float>{1, 2},
                  std::tuple<int, float>{2, 2},
              }));
}
TEST_F(TestCombineLatest, Transform)
{
    auto source1 = std::make_shared<node::TestSource<int>>();
    auto source2 = std::make_shared<node::TestSource<float>>();

    auto zip = std::make_shared<node::CombineLatestTransformComponent<std::tuple<int, float>, float>>(
        [](std::tuple<int, float>&& val) {
            return std::get<0>(val) + std::get<1>(val);
        });

    auto sink = std::make_shared<node::TestSink<float>>();

    mrc::make_edge(*source1, *zip->get_sink<0>());
    mrc::make_edge(*source2, *zip->get_sink<1>());
    mrc::make_edge(*zip, *sink);

    source1->run();
    source2->run();

    sink->run();
}

TEST_F(TestCombineLatest, CreateAndDestroy)
{
    {
        auto x = std::make_shared<node::CombineLatestComponent<std::tuple<int, float>>>();
    }

    {
        auto x = std::make_shared<node::CombineLatestTransformComponent<std::tuple<int, float>, float>>(
            [](std::tuple<int, float>&& val) {
                return std::get<0>(val) + std::get<1>(val);
            });
    }
}

}  // namespace mrc

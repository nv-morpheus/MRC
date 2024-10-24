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
#include "mrc/node/operators/with_latest_from.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mrc {

TEST_CLASS(WithLatestFrom);

TEST_F(TestWithLatestFrom, SourceToWLFToSink)
{
    auto source1 = std::make_shared<node::TestSource<int>>(5);
    auto source2 = std::make_shared<node::TestSource<float>>(5);
    auto source3 = std::make_shared<node::TestSource<std::string>>(std::vector<std::string>{"a", "b", "c", "d", "e"});

    auto with_latest = std::make_shared<node::WithLatestFromComponent<std::tuple<int, float, std::string>>>();

    auto sink = std::make_shared<node::TestSink<std::tuple<int, float, std::string>>>();

    mrc::make_edge(*source1, *with_latest->get_sink<0>());
    mrc::make_edge(*source2, *with_latest->get_sink<1>());
    mrc::make_edge(*source3, *with_latest->get_sink<2>());
    mrc::make_edge(*with_latest, *sink);

    // Push 2 from each
    source2->push(2);
    source1->push(2);
    source3->push(2);

    // Push 2 from each
    source2->push(2);
    source1->push(2);
    source3->push(2);

    // Push the rest
    source3->run();
    source1->run();
    source2->run();

    sink->run();

    EXPECT_EQ(sink->get_values(),
              (std::vector<std::tuple<int, float, std::string>>{
                  std::tuple<int, float, std::string>{0, 1, "a"},
                  std::tuple<int, float, std::string>{1, 1, "a"},
                  std::tuple<int, float, std::string>{2, 3, "b"},
                  std::tuple<int, float, std::string>{3, 3, "b"},
                  std::tuple<int, float, std::string>{4, 3, "e"},
              }));
}

TEST_F(TestWithLatestFrom, SourceToWLFToSinkComponent)
{
    auto source1 = std::make_shared<node::TestSource<int>>(5);
    auto source2 = std::make_shared<node::TestSource<float>>(5);
    auto source3 = std::make_shared<node::TestSource<std::string>>(std::vector<std::string>{"a", "b", "c", "d", "e"});

    auto with_latest = std::make_shared<node::WithLatestFromComponent<std::tuple<int, float, std::string>>>();

    auto sink = std::make_shared<node::TestSinkComponent<std::tuple<int, float, std::string>>>();

    mrc::make_edge(*source1, *with_latest->get_sink<0>());
    mrc::make_edge(*source2, *with_latest->get_sink<1>());
    mrc::make_edge(*source3, *with_latest->get_sink<2>());
    mrc::make_edge(*with_latest, *sink);

    // Push 2 from each
    source2->push(2);
    source1->push(2);
    source3->push(2);

    // Push 2 from each
    source2->push(2);
    source1->push(2);
    source3->push(2);

    // Push the rest
    source3->run();
    source1->run();
    source2->run();

    EXPECT_EQ(sink->get_values(),
              (std::vector<std::tuple<int, float, std::string>>{
                  std::tuple<int, float, std::string>{0, 1, "a"},
                  std::tuple<int, float, std::string>{1, 1, "a"},
                  std::tuple<int, float, std::string>{2, 3, "b"},
                  std::tuple<int, float, std::string>{3, 3, "b"},
                  std::tuple<int, float, std::string>{4, 3, "e"},
              }));
}

TEST_F(TestWithLatestFrom, UnevenPrimary)
{
    auto source1 = std::make_shared<node::TestSource<int>>(5);
    auto source2 = std::make_shared<node::TestSource<float>>(3);

    auto with_latest = std::make_shared<node::WithLatestFromComponent<std::tuple<int, float>>>();

    auto sink = std::make_shared<node::TestSink<std::tuple<int, float>>>();

    mrc::make_edge(*source1, *with_latest->get_sink<0>());
    mrc::make_edge(*source2, *with_latest->get_sink<1>());
    mrc::make_edge(*with_latest, *sink);

    source2->run();
    source1->run();

    sink->run();

    EXPECT_EQ(sink->get_values(),
              (std::vector<std::tuple<int, float>>{
                  std::tuple<int, float>{0, 2},
                  std::tuple<int, float>{1, 2},
                  std::tuple<int, float>{2, 2},
                  std::tuple<int, float>{3, 2},
                  std::tuple<int, float>{4, 2},
              }));
}

TEST_F(TestWithLatestFrom, UnevenSecondary)
{
    auto source1 = std::make_shared<node::TestSource<int>>(3);
    auto source2 = std::make_shared<node::TestSource<float>>(5);

    auto with_latest = std::make_shared<node::WithLatestFromComponent<std::tuple<int, float>>>();

    auto sink = std::make_shared<node::TestSink<std::tuple<int, float>>>();

    mrc::make_edge(*source1, *with_latest->get_sink<0>());
    mrc::make_edge(*source2, *with_latest->get_sink<1>());
    mrc::make_edge(*with_latest, *sink);

    source1->run();
    source2->run();

    sink->run();

    EXPECT_EQ(sink->get_values(),
              (std::vector<std::tuple<int, float>>{
                  std::tuple<int, float>{0, 0},
                  std::tuple<int, float>{1, 0},
                  std::tuple<int, float>{2, 0},
              }));
}

TEST_F(TestWithLatestFrom, TransformPrimaryFirst)
{
    auto source1 = std::make_shared<node::TestSource<int>>();
    auto source2 = std::make_shared<node::TestSource<float>>();

    auto zip = std::make_shared<node::WithLatestFromTransformComponent<std::tuple<int, float>, float>>(
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

    EXPECT_EQ(sink->get_values(), (std::vector<float>{0, 1, 2}));
}

TEST_F(TestWithLatestFrom, TransformSecondaryFirst)
{
    auto source1 = std::make_shared<node::TestSource<int>>();
    auto source2 = std::make_shared<node::TestSource<float>>();

    auto zip = std::make_shared<node::WithLatestFromTransformComponent<std::tuple<int, float>, float>>(
        [](std::tuple<int, float>&& val) {
            return std::get<0>(val) + std::get<1>(val);
        });

    auto sink = std::make_shared<node::TestSink<float>>();

    mrc::make_edge(*source1, *zip->get_sink<0>());
    mrc::make_edge(*source2, *zip->get_sink<1>());
    mrc::make_edge(*zip, *sink);

    source2->run();
    source1->run();

    sink->run();

    EXPECT_EQ(sink->get_values(), (std::vector<float>{2, 3, 4}));
}

TEST_F(TestWithLatestFrom, CreateAndDestroy)
{
    {
        auto x = std::make_shared<node::WithLatestFromComponent<std::tuple<int, float>>>();
    }

    {
        auto x = std::make_shared<node::WithLatestFromTransformComponent<std::tuple<int, float>, float>>(
            [](std::tuple<int, float>&& val) {
                return std::get<0>(val) + std::get<1>(val);
            });
    }
}

}  // namespace mrc

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/pipeline/pipeline.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <rxcpp/rx.hpp>

#include <cstddef>
#include <memory>
#include <ostream>
#include <random>
#include <set>
#include <string>
#include <utility>

namespace mrc {
class TestSegmentResources;
}  // namespace mrc

namespace mrc {

class StatGatherTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_int_distribution<> dist(10, 100);

        m_iterations = dist(generator);
        m_pipeline   = mrc::make_pipeline();

        auto init = [this](segment::IBuilder& segment) {
            auto src = segment.make_source<std::string>("src", [this](rxcpp::subscriber<std::string> s) {
                for (auto i = 0; i < m_iterations; ++i)
                {
                    s.on_next("One_" + std::to_string(i));
                }
                s.on_completed();
            });

            auto internal_1 = segment.make_node<std::string, std::string>("internal_1",
                                                                          rxcpp::operators::map([](std::string s) {
                                                                              return s + "-Mapped";
                                                                          }));

            segment.make_edge(src, internal_1);

            auto internal_2 = segment.make_node<std::string, std::string>("internal_2",
                                                                          rxcpp::operators::map([](std::string s) {
                                                                              return s;
                                                                          }));

            segment.make_edge(internal_1, internal_2);

            auto sink = segment.make_sink<std::string>(
                "sink",
                [](std::string x) {
                    VLOG(10) << x << std::endl;
                },
                []() {
                    VLOG(10) << "Completed" << std::endl;
                });

            segment.make_edge(internal_2, sink);
        };

        m_pipeline->make_segment("segment_stats_test", init);
    }

    void TearDown() override {}

    std::size_t m_iterations;
    std::unique_ptr<pipeline::IPipeline> m_pipeline;
    std::shared_ptr<TestSegmentResources> m_resources;
    std::set<std::string> m_components = {"src", "internal_1", "internal_2", "sink"};
};

}  // namespace mrc

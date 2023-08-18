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

#include "test_pymrc.hpp"

#include "pymrc/executor.hpp"
#include "pymrc/pipeline.hpp"

#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/pytypes.h>
#include <rxcpp/rx.hpp>

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace py    = pybind11;
namespace pymrc = mrc::pymrc;
using namespace std::string_literals;

PYMRC_TEST_CLASS(Executor);

TEST_F(TestExecutor, Execute)
{
    std::atomic<unsigned int> counter = 0;
    pymrc::Pipeline p;

    auto init = [&counter](mrc::segment::IBuilder& seg) {
        auto src = seg.make_source<bool>("src", [](rxcpp::subscriber<bool>& s) {
            if (s.is_subscribed())
            {
                s.on_next(true);
                s.on_next(false);
            }

            s.on_completed();
        });

        auto internal = seg.make_node<bool, unsigned int>("internal", rxcpp::operators::map([](bool b) {
                                                              unsigned int i{static_cast<unsigned int>(b)};
                                                              return i;
                                                          }));

        auto sink = seg.make_sink<unsigned int>("sink", [&counter](unsigned int x) {
            counter.fetch_add(x, std::memory_order_relaxed);
        });

        seg.make_edge(src, internal);
        seg.make_edge(internal, sink);
    };

    p.make_segment("seg1"s, init);
    p.make_segment("seg2"s, init);
    p.make_segment("seg3"s, init);

    auto options = std::make_shared<mrc::Options>();
    options->topology().user_cpuset("0");

    pymrc::Executor exec{options};
    exec.register_pipeline(p);

    exec.start();
    exec.join();

    EXPECT_EQ(counter, 3);
}

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

#include "test_pysrf.hpp"

#include "pysrf/pipeline.hpp"

#include "srf/channel/status.hpp"
#include "srf/core/executor.hpp"
#include "srf/engine/pipeline/ipipeline.hpp"
#include "srf/node/rx_node.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/node/rx_source.hpp"
#include "srf/options/options.hpp"
#include "srf/options/topology.hpp"
#include "srf/segment/builder.hpp"
#include "srf/segment/object.hpp"

#include <gtest/gtest.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <rxcpp/operators/rx-map.hpp>
#include <rxcpp/rx-includes.hpp>
#include <rxcpp/rx-observer.hpp>
#include <rxcpp/rx-operators.hpp>
#include <rxcpp/rx-predef.hpp>
#include <rxcpp/rx-subscriber.hpp>
#include <rxcpp/sources/rx-iterate.hpp>

#include <atomic>
#include <memory>
#include <string>

// IWYU thinks we need move & vector for auto internal = seg.make_rx_node
// IWYU pragma: no_include <utility>
// IWYU pragma: no_include <vector>

namespace py    = pybind11;
namespace pysrf = srf::pysrf;
using namespace std::string_literals;

PYSRF_TEST_CLASS(Pipeline);

// TEST_F(TestPipeline, Constructor)
// {
//     pysrf::Pipeline p;

//     // since the underlying pipeline is private swapping it
//     // out to see what the underlying pipeline was
//     auto pipe_ptr = p.swap();
//     EXPECT_NE(pipe_ptr, nullptr);

//     EXPECT_EQ(pipe_ptr->segment_count(), 0);
//     EXPECT_TRUE(pipe_ptr->sources().empty());
//     EXPECT_TRUE(pipe_ptr->sinks().empty());

//     EXPECT_THROW(pipe_ptr->lookup_id("fake"s), std::exception);
// }

// TEST_F(TestPipeline, MakeSegment)
// {
//     pysrf::Pipeline p;
//     p.make_segment("turtle"s, [](srf::segment::Builder& seg) {});
//     p.make_segment("lizard"s, [](srf::segment::Builder& seg) {});
//     p.make_segment("frog"s, [](srf::segment::Builder& seg) {});

//     auto pipe_ptr = p.swap();
//     EXPECT_EQ(pipe_ptr->segment_count(), 3);

//     auto segId = pipe_ptr->lookup_id("lizard"s);
//     EXPECT_EQ(pipe_ptr->lookup_name(segId), "lizard"s);
// }

TEST_F(TestPipeline, Execute)
{
    std::atomic<unsigned int> counter = 0;
    pysrf::Pipeline p;

    auto init = [&counter](srf::segment::Builder& seg) {
        auto src = seg.make_source<bool>("src", [](rxcpp::subscriber<bool>& s) {
            if (s.is_subscribed())
            {
                s.on_next(true);
                s.on_next(false);
            }

            s.on_completed();
        });

        auto internal = seg.make_node<bool, unsigned int>("internal", rxcpp::operators::map([](bool b) {
                                                              unsigned int i{b};
                                                              return i;
                                                          }));

        auto sink = seg.make_sink<unsigned int>(
            "sink", [&counter](unsigned int x) { counter.fetch_add(x, std::memory_order_relaxed); });

        seg.make_edge(src, internal);
        seg.make_edge(internal, sink);
    };

    p.make_segment("seg1"s, init);
    p.make_segment("seg2"s, init);
    p.make_segment("seg3"s, init);

    auto options = std::make_shared<srf::Options>();
    options->topology().user_cpuset("0");

    // note this is the base SRF executor not a pysrf executor
    srf::Executor exec{options};
    exec.register_pipeline(p.swap());

    py::gil_scoped_release release;
    exec.start();
    exec.join();

    EXPECT_EQ(counter, 3);
}

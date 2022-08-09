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

#include "pysrf/executor.hpp"
#include "pysrf/pipeline.hpp"

#include "srf/channel/status.hpp"
#include "srf/node/rx_node.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/node/rx_source.hpp"
#include "srf/options/options.hpp"
#include "srf/options/topology.hpp"
#include "srf/segment/builder.hpp"
#include "srf/segment/object.hpp"

#include <boost/hana/if.hpp>
#include <gtest/gtest.h>     // IWYU pragma: keep
#include <pybind11/embed.h>  // IWYU pragma: keep
#include <rxcpp/operators/rx-map.hpp>
#include <rxcpp/rx.hpp>

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// IWYU pragma: no_include "gtest/gtest_pred_impl.h"
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include "rxcpp/sources/rx-iterate.hpp"
// IWYU pragma: no_include "rx-includes.hpp"

namespace py    = pybind11;
namespace pysrf = srf::pysrf;
using namespace std::string_literals;

PYSRF_TEST_CLASS(Executor);

TEST_F(TestExecutor, Execute)
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

    pysrf::Executor exec{options};
    exec.register_pipeline(p);

    exec.start();
    exec.join();

    EXPECT_EQ(counter, 3);
}

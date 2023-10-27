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

#include "pymrc/asyncio_runnable.hpp"
#include "pymrc/executor.hpp"
#include "pymrc/pipeline.hpp"
#include "pymrc/port_builders.hpp"
#include "pymrc/types.hpp"

#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_sink_base.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/node/rx_source_base.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep
#include <rxcpp/rx.hpp>

#include <atomic>
#include <cstddef>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace py    = pybind11;
namespace pymrc = mrc::pymrc;
using namespace std::string_literals;
using namespace py::literals;

PYMRC_TEST_CLASS(AsyncioRunnable);

class MyAsyncioRunnable : public pymrc::AsyncioRunnable<int, unsigned int>
{
    mrc::coroutines::AsyncGenerator<unsigned int> on_data(int&& value) override
    {
        co_yield value;
        co_yield value * 2;
    };
};

TEST_F(TestAsyncioRunnable, Execute)
{
    std::atomic<unsigned int> counter = 0;
    pymrc::Pipeline p;

    pybind11::module_::import("mrc.core.coro");

    auto init = [&counter](mrc::segment::IBuilder& seg) {
        auto src = seg.make_source<int>("src", [](rxcpp::subscriber<int>& s) {
            if (s.is_subscribed())
            {
                s.on_next(5);
                s.on_next(10);
                s.on_next(7);
            }

            s.on_completed();
        });

        auto internal = seg.construct_object<MyAsyncioRunnable>("internal");

        auto sink = seg.make_sink<unsigned int>("sink", [&counter](unsigned int x) {
            counter.fetch_add(x, std::memory_order_relaxed);
        });

        seg.make_edge(src, internal);
        seg.make_edge(internal, sink);
    };

    p.make_segment("seg1"s, init);

    auto options = std::make_shared<mrc::Options>();
    options->topology().user_cpuset("0");

    pymrc::Executor exec{options};
    exec.register_pipeline(p);

    exec.start();
    exec.join();

    EXPECT_EQ(counter, 66);
}

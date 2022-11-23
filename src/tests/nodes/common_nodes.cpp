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

#include "common_nodes.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/memory/literals.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/runnable/context.hpp"

#include <boost/fiber/operations.hpp>
#include <glog/logging.h>
#include <rxcpp/rx.hpp>
#include <rxcpp/sources/rx-iterate.hpp>

#include <chrono>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace mrc;
using namespace mrc::memory::literals;

namespace test::nodes {

std::unique_ptr<node::RxSink<int>> int_sink()
{
    return std::make_unique<node::RxSink<int>>(
        [](int x) { VLOG(1) << runnable::Context::get_runtime_context().info() << ": data=" << x; });
}

std::unique_ptr<node::RxSink<int>> int_sink_throw_on_even()
{
    return std::make_unique<node::RxSink<int>>([](int x) {
        VLOG(1) << runnable::Context::get_runtime_context().info() << ": data=" << x;
        if (x % 2 == 0)
        {
            throw std::runtime_error("odds only");
        }
    });
}

std::unique_ptr<node::RxSource<int>> infinite_int_rx_source()
{
    return std::make_unique<node::RxSource<int>>(rxcpp::observable<>::create<int>([](rxcpp::subscriber<int> s) {
        int i = 1;
        while (s.is_subscribed())
        {
            VLOG(1) << runnable::Context::get_runtime_context().info() << "; emitting " << i;
            s.on_next(i++);
            boost::this_fiber::sleep_for(std::chrono::milliseconds(10));
        }
        s.on_completed();
    }));
}

std::unique_ptr<node::RxSource<int>> finite_int_rx_source(int count)
{
    return std::make_unique<node::RxSource<int>>(rxcpp::observable<>::create<int>([count](rxcpp::subscriber<int> s) {
        VLOG(1) << runnable::Context::get_runtime_context().info();
        for (int i = 0; i < count; i++)
        {
            s.on_next(i);
        }
        s.on_completed();
    }));
}

std::unique_ptr<BufferSource<1_MiB>> infinte_buffer_source(std::size_t capacity)
{
    return std::make_unique<BufferSource<1_MiB>>(capacity);
}

}  // namespace test::nodes

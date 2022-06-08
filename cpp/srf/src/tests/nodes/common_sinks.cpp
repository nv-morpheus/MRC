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

#include "srf/channel/status.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/runnable/context.hpp"

#include <glog/logging.h>
#include "rxcpp/rx-includes.hpp"
#include "rxcpp/rx-observer.hpp"
#include "rxcpp/rx-operators.hpp"
#include "rxcpp/rx-predef.hpp"
#include "rxcpp/rx-subscriber.hpp"

#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace srf;

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

}  // namespace test::nodes

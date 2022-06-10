/**
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <glog/logging.h>

#include <srf/node/rx_sink.hpp>
#include <srf/pipeline/pipeline.hpp>
#include <srf/srf.hpp>

#include "sources.hpp"

using namespace srf;

int main(int argc, char* argv[])
{
    std::atomic<long> counter = 0;

    // srf options
    auto options = std::make_unique<srf::Options>();

    // create executor
    Executor executor(std::move(options));

    // create pipeline object
    auto pipeline = pipeline::make_pipeline();

    // create a segment - a pipeline can consist of multiple segments
    auto seg = segment::Definition::create("quickstart", [&counter](segment::Builder& s) {
        // this is where data is produced - from a file, from the network, from a sensor, etc.
        auto src = s.make_source<int>("int_source", [](rxcpp::subscriber<int> s) {
            s.on_next(1);
            s.on_next(2);
            s.on_next(3);
            s.on_completed();
        });

        auto sink =
            s.make_sink<int>("int_sink", rxcpp::make_observer_dynamic<int>([&counter](int data) { counter++; }));

        s.make_edge(src, sink);
    });

    // register segments with the pipeline
    pipeline->register_segment(seg);

    // register the pipeline with the executor
    executor.register_pipeline(std::move(pipeline));

    // start the pipeline and wait until it finishes
    std::cout << "srf pipeline starting..." << std::endl;
    executor.start();
    executor.join();
    std::cout << "srf pipeline complete: counter should be 3; counter=" << counter << std::endl;

    return 0;
};

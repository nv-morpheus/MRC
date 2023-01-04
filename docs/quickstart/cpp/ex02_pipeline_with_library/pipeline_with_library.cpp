/**
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nodes.hpp"

#include <glog/logging.h>
#include <mrc/mrc.hpp>
#include <mrc/node/rx_sink.hpp>
#include <mrc/pipeline/pipeline.hpp>

using namespace mrc;
using namespace mrc::quickstart::cpp::common;

int main(int argc, char* argv[])
{
    std::atomic<long> counter = 0;

    // mrc options
    auto options = std::make_unique<mrc::Options>();

    // create executor
    Executor executor(std::move(options));

    // create pipeline object
    auto pipeline = pipeline::make_pipeline();

    // create a segment - a pipeline can consist of multiple segments; however in this example we will use only one
    auto seg = segment::Definition::create("quickstart", [&counter](segment::Builder& s) {
        // Source
        // This first "node" is a source node which has no upstream dependencies. It is responsible for producing data
        // to be consume by downstream nodes
        // Here we are constructing the IntSource from the libmrc_quicstart library
        auto source = s.construct_object<IntSource>("int_source");

        // Node
        // A Node is both a Source and a Sink, it connects to an upstream provider/source and a downstream
        // subscriber/sink. This examples accects an upstream int and provides a downstream float which is the input
        // value scaled by 2.5.
        auto node = s.make_node<int, float>("int_x2_to_float",
                                            rxcpp::operators::map([](const int& data) { return float(2.5F * data); }));

        // Sink
        // Sinks are terminators. They only accept upstream connections and do not provide the ability to pass data on.
        // This sink increments a captured counter and outputs the values
        auto sink = s.make_sink<float>("float_sink", rxcpp::make_observer_dynamic<float>([&counter](float data) {
                                           counter++;
                                           std::cout << "sink: " << data << std::endl;
                                       }));

        // Edges
        // We definite the connects between nodes by forming edges. Edges allow data to propagate in between nodes using
        // a policy defined by Channel objects. The default channel is a yielding buffered channel which yields upstream
        // writers when the channel is full of data.

        // source -> node
        s.make_edge(source, node);

        // node -> sink
        s.make_edge(node, sink);
    });

    // register segments with the pipeline
    pipeline->register_segment(seg);

    // register the pipeline with the executor
    executor.register_pipeline(std::move(pipeline));

    // start the pipeline and wait until it finishes
    std::cout << "mrc pipeline starting..." << std::endl;
    executor.start();
    executor.join();
    std::cout << "mrc pipeline complete: counter should be 3; counter=" << counter << std::endl;

    return 0;
};

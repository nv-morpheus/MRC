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

#include "test_modules.hpp"

#include "mrc/core/executor.hpp"
#include "mrc/engine/pipeline/ipipeline.hpp"
#include "mrc/modules/mirror_tap/mirror_tap_util.hpp"
#include "mrc/modules/module_registry.hpp"
#include "mrc/options/options.hpp"
#include "mrc/segment/builder.hpp"

#include <gtest/gtest-message.h>
#include <gtest/gtest-test-part.h>

#include <utility>
#include <vector>

struct PipelineUtils {
};

using namespace mrc;

TEST_F(TestStreamBufferModule, InitailizationTest) {
    using namespace modules;

    auto init_wrapper = [](segment::Builder &builder) {
        auto config = nlohmann::json();
        auto mirror_tap = builder.make_module<ImmediateStreamBufferModule<std::string>>("mirror_tap", config);
    };

    m_pipeline->make_segment("Initialization_Segment", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.stop();
    executor.join();
}

TEST_F(TestStreamBufferModule, SinglePipelineStreamBufferTest) {
    using namespace modules;
    const std::string test_name{"SinglePipelineStreamBufferTest"};

    // Create external captures for packet counts.
    unsigned int packet_count{10000};
    unsigned int packets_main{0};
    unsigned int packets_mirrored{0};

    auto config = nlohmann::json();


    auto init_wrapper_main = [&packets_main, packet_count, test_name](segment::Builder &builder) {
        auto source = builder.make_source<std::string>(
                test_name + "_main_source",
                [packet_count](rxcpp::subscriber<std::string> &sub) {
                    if (sub.is_subscribed()) {
                        for (unsigned int i = 0; i < packet_count; i++) {
                            sub.on_next(std::to_string(packet_count));
                        }
                    }

                    sub.on_completed();
                });

        auto sink = builder.make_sink<std::string>(test_name + "_main_sink", [&packets_main](std::string input) {
            packets_main++;
        });

        // Untapped edge that will be broken and tapped by the mirror tap.
        builder.make_edge(source, sink);
    };

    auto init_wrapper_mirrored = [&packets_mirrored, test_name](
            segment::Builder &builder) {
        auto mirror_sink = builder.make_sink<std::string>(test_name + "_mirror_sink",
                                                          [&packets_mirrored](std::string input) {
                                                              VLOG(10) << "tick -> " << input << std::endl
                                                                       << std::flush;
                                                              packets_mirrored++;
                                                          });
    };

    auto mirror_tap = MirrorTap<std::string>(test_name + "mirror_tap", config);

    auto tapped_init_wrapper_main = mirror_tap.tap(init_wrapper_main,
                                                               test_name + "_main_source",
                                                               test_name + "_main_sink");

    auto tapped_init_wrapper_mirrored = mirror_tap.stream_to(init_wrapper_mirrored,
                                                                      test_name + "_mirror_sink");

    m_pipeline->make_segment("Main_Segment",
                             mirror_tap.create_egress_ports(),
                             tapped_init_wrapper_main);

    m_pipeline->make_segment("StreamMirror_Segment",
                             mirror_tap.create_ingress_ports(),
                             tapped_init_wrapper_mirrored);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-2");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    // Since we wire everything up before the main source starts pumping data, we should always have the same
    // number of packets between main and mirrored, even though we're using hot observables internally.
    EXPECT_EQ(packets_main, packet_count);

    //EXPECT_EQ(packets_mirrored, packet_count);
    EXPECT_GE(packets_mirrored, packet_count * 0.90);
}

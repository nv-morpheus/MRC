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
#include "mrc/modules/mirror_tap/mirror_tap_module.hpp"
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

TEST_F(TestMirrorTapModule, ConstructorTest) {
    using namespace modules;

    auto config = nlohmann::json();

    auto mod1 = MirrorTapModule<std::string>("mirror_tap", config);
}

TEST_F(TestMirrorTapModule, InitailizationTest) {
    using namespace modules;

    auto init_wrapper = [](segment::Builder &builder) {
        auto config = nlohmann::json();
        auto mirror_tap = builder.make_module<MirrorTapModule<std::string>>("mirror_tap", config);
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

TEST_F(TestMirrorTapModule, SinglePipelineMirrorTapTest) {
    using namespace modules;
    const std::string test_name{"SinglePipelineMirrorTapTest"};

    // Create external captures for packet counts.
    unsigned int packets_main{0};
    unsigned int packets_mirrored{0};

    auto config = nlohmann::json();

    auto mirror_tap = std::make_shared<MirrorTapModule<std::string>>(test_name + "_mirror_tap", config);
    auto init_wrapper_main = [&packets_main, &mirror_tap, &test_name](segment::Builder &builder) {
        builder.init_module(mirror_tap);

        auto source = builder.make_source<std::string>(test_name + "_main_source",
                                                       [](rxcpp::subscriber<std::string> &sub) {
                                                           if (sub.is_subscribed()) {
                                                               sub.on_next("one");
                                                               sub.on_next("two");
                                                               sub.on_next("three");
                                                               sub.on_next("four");
                                                           }

                                                           sub.on_completed();
                                                       });

        // mirror tap has an input and output port, and will create an egress port that can be attached to.
        builder.make_edge(source, mirror_tap->input_port("input"));

        auto sink = builder.make_sink<std::string>(test_name + "_main_sink", [&packets_main](std::string input) {
            packets_main++;
        });

        builder.make_edge(mirror_tap->output_port("output"), sink);
    };

    auto init_wrapper_mirrored = [&packets_mirrored, &mirror_tap, &test_name](segment::Builder &builder) {
        auto mirror_ingress = builder.get_ingress<std::string>(mirror_tap->get_port_name());
        auto mirror_sink = builder.make_sink<std::string>(test_name + "_mirror_sink",
                                                          [&packets_mirrored](std::string input) {
                                                              packets_mirrored++;
                                                          });

        builder.make_edge(mirror_ingress, mirror_sink);

    };

    m_pipeline->make_segment("Main_Segment",
                             mirror_tap->create_egress_ports(),
                             init_wrapper_main);

    m_pipeline->make_segment("Mirror_Segment",
                             mirror_tap->create_ingress_ports(),
                             init_wrapper_mirrored);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(packets_main, 4);
    EXPECT_EQ(packets_mirrored, 4);
}
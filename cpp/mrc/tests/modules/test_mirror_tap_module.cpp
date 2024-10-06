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

#include "test_modules.hpp"

#include "mrc/cuda/device_guard.hpp"
#include "mrc/experimental/modules/mirror_tap/mirror_tap.hpp"
#include "mrc/modules/properties/persistent.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/node/rx_source_base.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/pipeline/executor.hpp"
#include "mrc/pipeline/pipeline.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/egress_ports.hpp"
#include "mrc/segment/ingress_ports.hpp"
#include "mrc/segment/object.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <rxcpp/rx.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

using namespace mrc;

TEST_F(TestMirrorTapModule, ConstructorTest)
{
    using namespace modules;

    auto config = nlohmann::json();

    auto mod1 = MirrorTapModule<std::string>("mirror_tap", config);
}

TEST_F(TestMirrorTapModule, PortNamingTest)
{
    // Test default constructor
    mrc::modules::MirrorTapModule<int> module1{"module1"};
    EXPECT_EQ(module1.tap_egress_port_name(), "mirror_tap_source_0");

    // Test constructor with config
    nlohmann::json config1;
    mrc::modules::MirrorTapModule<int> module2{"module2", config1};
    EXPECT_EQ(module2.tap_egress_port_name(), "mirror_tap_source_1");

    // Test constructor with config
    nlohmann::json config2{{"tap_id_override", "mirror_tap_override"}};
    mrc::modules::MirrorTapModule<int> module3{"module3", config2};
    EXPECT_EQ(module3.tap_egress_port_name(), "mirror_tap_override");
}

TEST_F(TestMirrorTapModule, InitailizationTest)
{
    using namespace modules;
    const std::string test_name{"InitializationTest"};

    auto config = nlohmann::json();

    auto mirror_tap        = std::make_shared<MirrorTapModule<std::string>>(test_name + "_mirror_tap", config);
    auto init_wrapper_main = [&mirror_tap, &test_name](segment::IBuilder& builder) {
        builder.init_module(mirror_tap);

        auto source = builder.make_source<std::string>(test_name + "_main_source",
                                                       [](rxcpp::subscriber<std::string>& sub) {});

        // mirror tap has an input and output port, and will create an egress port that can be attached to.
        builder.make_edge(source, mirror_tap->input_port("input"));

        auto sink = builder.make_sink<std::string>(test_name + "_main_sink", [](std::string input) {});

        builder.make_edge(mirror_tap->output_port("output"), sink);
    };

    auto init_wrapper_mirrored = [&mirror_tap, &test_name](segment::IBuilder& builder) {
        auto mirror_ingress = builder.get_ingress<std::string>(mirror_tap->tap_egress_port_name());
        auto mirror_sink    = builder.make_sink<std::string>(test_name + "_mirror_sink", [](std::string input) {});

        builder.make_edge(mirror_ingress, mirror_sink);
    };

    m_pipeline->make_segment("Main_Segment",
                             segment::EgressPorts<std::string>({mirror_tap->tap_egress_port_name()}),
                             init_wrapper_main);

    m_pipeline->make_segment("Mirror_Segment",
                             segment::IngressPorts<std::string>({mirror_tap->tap_egress_port_name()}),
                             init_wrapper_mirrored);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(mirror_tap->input_ports().size(), 1U);
    EXPECT_EQ(mirror_tap->output_ports().size(), 1U);
}

TEST_F(TestMirrorTapModule, SinglePipelineMirrorTapTest)
{
    using namespace modules;
    const std::string test_name{"SinglePipelineMirrorTapTest"};

    // Create external captures for packet counts.
    unsigned int packets_main{0};
    unsigned int packets_mirrored{0};

    auto config = nlohmann::json();

    auto mirror_tap        = std::make_shared<MirrorTapModule<std::string>>(test_name + "_mirror_tap", config);
    auto init_wrapper_main = [&packets_main, &mirror_tap, &test_name](segment::IBuilder& builder) {
        builder.init_module(mirror_tap);

        auto source = builder.make_source<std::string>(test_name + "_main_source",
                                                       [](rxcpp::subscriber<std::string>& sub) {
                                                           if (sub.is_subscribed())
                                                           {
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

    auto init_wrapper_mirrored = [&packets_mirrored, &mirror_tap, &test_name](segment::IBuilder& builder) {
        auto mirror_ingress = builder.get_ingress<std::string>(mirror_tap->tap_egress_port_name());
        auto mirror_sink    = builder.make_sink<std::string>(test_name + "_mirror_sink",
                                                          [&packets_mirrored](std::string input) {
                                                              packets_mirrored++;
                                                          });

        builder.make_edge(mirror_ingress, mirror_sink);
    };

    m_pipeline->make_segment("Main_Segment",
                             segment::EgressPorts<std::string>({mirror_tap->tap_egress_port_name()}),
                             init_wrapper_main);

    m_pipeline->make_segment("Mirror_Segment",
                             segment::IngressPorts<std::string>({mirror_tap->tap_egress_port_name()}),
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

/**
 * Test this configuration:
 *
 * Segment 0: [ Source ] -- [ MirrorTap1 ] -- [ internal ] -- [ MirrorTap2 ] --[ Sink ]
 *                                        \                                 \
 * Segment 1:                              -  [ MirrorSink1]                 \
 * Segment 2:                                                                 - [ MirrorSink2 ]
 */
TEST_F(TestMirrorTapModule, SinglePipelineMultiInlineMirrorTapTest)
{
    using namespace modules;
    const std::string test_name{"SinglePipelineMultiInlineMirrorTapTest"};

    // Create external captures for packet counts.
    unsigned int packets_main{0};

    auto config = nlohmann::json();

    auto mirror_tap_one = std::make_shared<MirrorTapModule<std::string>>(test_name + "_mirror_tap_one", config);
    auto mirror_tap_two = std::make_shared<MirrorTapModule<std::string>>(test_name + "_mirror_tap_two", config);

    auto init_wrapper_main = [&packets_main, &mirror_tap_one, &mirror_tap_two, &test_name](segment::IBuilder& builder) {
        builder.init_module(mirror_tap_one);
        builder.init_module(mirror_tap_two);

        auto source = builder.make_source<std::string>(test_name + "_main_source",
                                                       [](rxcpp::subscriber<std::string>& sub) {
                                                           if (sub.is_subscribed())
                                                           {
                                                               sub.on_next("one");
                                                               sub.on_next("two");
                                                               sub.on_next("three");
                                                               sub.on_next("four");
                                                           }

                                                           sub.on_completed();
                                                       });

        // mirror tap has an input and output port, and will create an egress port that can be attached to.
        builder.make_edge(source, mirror_tap_one->input_port("input"));

        auto internal = builder.make_node<std::string>(test_name + "_internal",
                                                       rxcpp::operators::map([](std::string input) {
                                                           return input;
                                                       }));

        builder.make_edge(mirror_tap_one->output_port("output"), internal);
        builder.make_edge(internal, mirror_tap_two->input_port("input"));

        auto sink = builder.make_sink<std::string>(test_name + "_main_sink", [&packets_main](std::string input) {
            packets_main++;
        });

        builder.make_edge(mirror_tap_two->output_port("output"), sink);
    };

    auto multi_sink_mirror_one     = std::make_shared<mrc::MultiSinkModule<std::string, 1>>(test_name +
                                                                                        "_multi_sink_mirror_1");
    auto init_wrapper_mirrored_one = [&mirror_tap_one, &multi_sink_mirror_one, test_name](segment::IBuilder& builder) {
        auto mirror_ingress_one = builder.get_ingress<std::string>(mirror_tap_one->tap_egress_port_name());

        builder.init_module(multi_sink_mirror_one);
        builder.make_edge(mirror_ingress_one, multi_sink_mirror_one->input_port("input_0"));
    };

    auto multi_sink_mirror_two     = std::make_shared<mrc::MultiSinkModule<std::string, 1>>(test_name +
                                                                                        "_multi_sink_mirror_2");
    auto init_wrapper_mirrored_two = [&mirror_tap_two, &multi_sink_mirror_two, test_name](segment::IBuilder& builder) {
        auto mirror_ingress_two = builder.get_ingress<std::string>(mirror_tap_two->tap_egress_port_name());

        builder.init_module(multi_sink_mirror_two);
        builder.make_edge(mirror_ingress_two, multi_sink_mirror_two->input_port("input_0"));
    };

    m_pipeline->make_segment("Main_Segment",
                             segment::EgressPorts<std::string, std::string>(
                                 {mirror_tap_one->tap_egress_port_name(), mirror_tap_two->tap_egress_port_name()}),
                             init_wrapper_main);

    m_pipeline->make_segment("Mirror_Segment_1",
                             segment::IngressPorts<std::string>({mirror_tap_one->tap_egress_port_name()}),
                             init_wrapper_mirrored_one);

    m_pipeline->make_segment("Mirror_Segment_2",
                             segment::IngressPorts<std::string>({mirror_tap_two->tap_egress_port_name()}),
                             init_wrapper_mirrored_two);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(packets_main, 4);
    EXPECT_EQ(multi_sink_mirror_one->get_received(0), 4);
    EXPECT_EQ(multi_sink_mirror_two->get_received(0), 4);
}

/**
 * Test this configuration:
 *
 * Segment 1: [ Source1 ] -- [ MirrorTap1 ] -- [ Sink1 ]
 *                                         \
 * Segment 2:                               -- [ MirrorSink1]
 * Segment 1: [ Source2 ] -- [ MirrorTap2 ] -- [ Sink2 ]
 *                                         \
 * Segment 2:                               -- [ MirrorSink2]
 * Segment 1: [ Source3 ] -- [ MirrorTap3 ] -- [ Sink3 ]
 *                                         \
 * Segment 2:                               -- [ MirrorSink3]
 */
TEST_F(TestMirrorTapModule, SinglePipelineMultiMirrorTapTest)
{
    using namespace modules;
    const std::string test_name{"SinglePipelineMultiMirrorTapTest"};

    auto config = nlohmann::json();

    auto mirror_tap_one    = std::make_shared<MirrorTapModule<std::string>>(test_name + "_mirror_tap_one", config);
    auto mirror_tap_two    = std::make_shared<MirrorTapModule<std::string>>(test_name + "_mirror_tap_two", config);
    auto mirror_tap_three  = std::make_shared<MirrorTapModule<std::string>>(test_name + "_mirror_tap_three", config);
    auto multi_sink_main   = std::make_shared<MultiSinkModule<std::string, 3>>(test_name + "_multi_sink_main");
    auto init_wrapper_main = [&mirror_tap_one, &mirror_tap_two, &mirror_tap_three, &multi_sink_main, &test_name](
                                 segment::IBuilder& builder) {
        builder.init_module(mirror_tap_one);
        builder.init_module(mirror_tap_two);
        builder.init_module(mirror_tap_three);

        auto multi_source_mod = builder.make_module<mrc::MultiSourceModule<std::string, 3, 4>>(test_name +
                                                                                               "_multi_source_mod");

        builder.make_edge<std::string>(multi_source_mod->output_port("output_0"), mirror_tap_one->input_port("input"));
        builder.make_edge<std::string>(multi_source_mod->output_port("output_1"), mirror_tap_two->input_port("input"));
        builder.make_edge<std::string>(multi_source_mod->output_port("output_2"),
                                       mirror_tap_three->input_port("input"));

        builder.init_module(multi_sink_main);

        builder.make_edge<std::string>(mirror_tap_one->output_port("output"), multi_sink_main->input_port("input_0"));
        builder.make_edge<std::string>(mirror_tap_two->output_port("output"), multi_sink_main->input_port("input_1"));
        builder.make_edge<std::string>(mirror_tap_three->output_port("output"), multi_sink_main->input_port("input_2"));
    };

    auto multi_sink_mirror = std::make_shared<mrc::MultiSinkModule<std::string, 3>>(test_name + "_multi_sink_mirror");
    auto init_wrapper_mirrored =
        [&mirror_tap_one, &mirror_tap_two, &mirror_tap_three, &multi_sink_mirror](segment::IBuilder& builder) {
            auto mirror_ingress_one   = builder.get_ingress<std::string>(mirror_tap_one->tap_egress_port_name());
            auto mirror_ingress_two   = builder.get_ingress<std::string>(mirror_tap_two->tap_egress_port_name());
            auto mirror_ingress_three = builder.get_ingress<std::string>(mirror_tap_three->tap_egress_port_name());

            builder.init_module(multi_sink_mirror);

            builder.make_edge<std::string>(mirror_ingress_one, multi_sink_mirror->input_port("input_0"));
            builder.make_edge<std::string>(mirror_ingress_two, multi_sink_mirror->input_port("input_1"));
            builder.make_edge<std::string>(mirror_ingress_three, multi_sink_mirror->input_port("input_2"));
        };

    m_pipeline->make_segment(
        "Main_Segment",
        segment::EgressPorts<std::string, std::string, std::string>({mirror_tap_one->tap_egress_port_name(),
                                                                     mirror_tap_two->tap_egress_port_name(),
                                                                     mirror_tap_three->tap_egress_port_name()}),
        init_wrapper_main);

    m_pipeline->make_segment(
        "Mirror_Segment",
        segment::IngressPorts<std::string, std::string, std::string>({mirror_tap_one->tap_egress_port_name(),
                                                                      mirror_tap_two->tap_egress_port_name(),
                                                                      mirror_tap_three->tap_egress_port_name()}),
        init_wrapper_mirrored);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    for (std::size_t i = 0; i < 3; i++)
    {
        EXPECT_EQ(multi_sink_main->get_received(i), 4);
        EXPECT_EQ(multi_sink_mirror->get_received(i), 4);
    }
}

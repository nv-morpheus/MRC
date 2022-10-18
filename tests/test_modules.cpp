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

#include "test_segment.hpp"

#include "srf/core/executor.hpp"
#include "srf/engine/pipeline/ipipeline.hpp"
#include "srf/options/options.hpp"
#include "srf/segment/builder.hpp"

#include <gtest/gtest-message.h>
#include <gtest/gtest-test-part.h>
#include <rxcpp/rx-subscriber.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

TEST_F(SegmentTests, InitModuleTest)
{
    using namespace modules;

    auto config_1            = nlohmann::json();
    auto config_2            = nlohmann::json();
    config_2["config_key_1"] = true;

    auto mod1 = SimpleModule("InitModuleTest_mod1");
    auto mod2 = ConfigurableModule("InitModuleTest_2");
    auto mod3 = ConfigurableModule("InitModuleTest_3", config_1);
    auto mod4 = ConfigurableModule("InitModuleTest_4", config_2);

    auto mod1_inputs = mod1.input_ids();

    ASSERT_EQ(mod1_inputs.size(), 2);

    auto mod2_inputs  = mod2.input_ids();
    auto mod2_outputs = mod2.output_ids();

    ASSERT_EQ(mod2_inputs.size(), 1);
    ASSERT_EQ(mod2_outputs.size(), 1);

    ASSERT_EQ(mod4.config().contains("config_key_1"), true);
}

TEST_F(SegmentTests, EndToEndTest)
{
    using namespace modules;
    unsigned int packets_1{0};
    unsigned int packets_2{0};
    unsigned int packets_3{0};

    auto init_wrapper = [&packets_1, &packets_2, &packets_3](segment::Builder& builder) {
        auto simple_mod       = builder.make_module<SimpleModule>("EndToEndTest_mod1");
        auto configurable_mod = builder.make_module<ConfigurableModule>("EndToEndTest_mod2");

        auto source1 = builder.make_source<bool>("src1", [](rxcpp::subscriber<bool>& sub) {
            if (sub.is_subscribed())
            {
                sub.on_next(true);
                sub.on_next(false);
                sub.on_next(true);
                sub.on_next(true);
            }

            sub.on_completed();
        });

        // Ex1. Partially dynamic edge construction
        builder.make_edge(source1, simple_mod.input_ports("input1"));

        auto source2 = builder.make_source<bool>("src2", [](rxcpp::subscriber<bool>& sub) {
            if (sub.is_subscribed())
            {
                sub.on_next(true);
                sub.on_next(false);
                sub.on_next(false);
                sub.on_next(false);
                sub.on_next(true);
                sub.on_next(false);
            }

            sub.on_completed();
        });

        // Ex2. Dynamic edge construction -- requires type specification
        builder.make_dynamic_edge<bool, bool>(source2, simple_mod.input_ports("input2"));

        auto sink1 = builder.make_sink<std::string>("sink1", [&packets_1](std::string input) {
            packets_1++;
            VLOG(10) << "Sinking " << input << std::endl;
        });

        builder.make_edge(simple_mod.output_ports("output1"), sink1);

        auto sink2 = builder.make_sink<std::string>("sink2", [&packets_2](std::string input) {
            packets_2++;
            VLOG(10) << "Sinking " << input << std::endl;
        });

        builder.make_edge(simple_mod.output_ports("output2"), sink2);

        auto source3 = builder.make_source<bool>("src3", [](rxcpp::subscriber<bool>& sub) {
            if (sub.is_subscribed())
            {
                sub.on_next(true);
                sub.on_next(false);
                sub.on_next(true);
                sub.on_next(true);
            }

            sub.on_completed();
        });

        builder.make_edge(source3, configurable_mod.input_ports("configurable_input_a"));

        auto sink3 = builder.make_sink<std::string>("sink3", [&packets_3](std::string input) {
            packets_3++;
            VLOG(10) << "Sinking " << input << std::endl;
        });

        builder.make_edge(configurable_mod.output_ports("configurable_output_x"), sink3);
    };

    m_pipeline->make_segment("SimpleModule_Segment", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(packets_1, 4);
    EXPECT_EQ(packets_2, 6);
    EXPECT_EQ(packets_3, 4);
}
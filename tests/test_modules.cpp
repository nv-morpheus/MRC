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

#include "test_segment.hpp"
#include "test_modules.hpp"

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


TEST_F(SegmentTests, InitModuleTest) {
    using namespace modules;

    auto config_1 = nlohmann::json();
    auto config_2 = nlohmann::json();
    config_2["config_key_1"] = true;

    auto mod1 = SimpleModule("module_1");
    auto mod2 = ConfigurableModule("module_2");
    auto mod3 = ConfigurableModule("module_3", config_1);
    auto mod4 = ConfigurableModule("module_4", config_2);

    auto mymod_inputs = mod1.input_ids();

    ASSERT_EQ(mymod_inputs.size(), 2);

    auto othermod_inputs = mod2.input_ids();
    auto othermod_outputs = mod2.output_ids();

    ASSERT_EQ(othermod_inputs.size(), 1);
    ASSERT_EQ(othermod_outputs.size(), 1);

    ASSERT_EQ(mod3.m_was_configured, false);
    ASSERT_EQ(mod4.m_was_configured, true);
}

TEST_F(SegmentTests, BasicModuleTest)
{
    using namespace modules;
    auto init_wrapper = [](segment::Builder& builder) {
        auto my_mod = builder.make_module<SimpleModule>("Module1");
        auto my_mod_other = builder.make_module<ConfigurableModule>("Module2");

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
        builder.make_edge(source1, my_mod.input_ports("input1"));

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
        builder.make_dynamic_edge<bool, bool>(source2, my_mod.input_ports("input2"));

        auto sink1 = builder.make_sink<std::string>("sink1", [](std::string input){
            std::cout << "Sinking " << input << std::endl;
        });

        builder.make_edge(my_mod.output_ports("output1"), sink1);


        auto sink2 = builder.make_sink<std::string>("sink2", [](std::string input){
          std::cout << "Sinking " << input << std::endl;
        });

        builder.make_edge(my_mod.output_ports("output2"), sink2);
    };

    m_pipeline->make_segment("MyModule_Segment", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();
}
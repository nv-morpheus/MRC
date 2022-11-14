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

#include "srf/core/executor.hpp"
#include "srf/engine/pipeline/ipipeline.hpp"
#include "srf/modules/mirror_tap/mirror_tap_module.hpp"
#include "srf/modules/module_registry.hpp"
#include "srf/options/options.hpp"
#include "srf/segment/builder.hpp"

#include <gtest/gtest-message.h>
#include <gtest/gtest-test-part.h>

#include <utility>
#include <vector>

TEST_F(TestMirrorTapModule, ConstructorTest)
{
    using namespace modules;

    auto config = nlohmann::json();

    auto mod1 = MirrorTapModule<std::string>("mirror_tap", config);
}

TEST_F(TestMirrorTapModule, InitailizationTest)
{
    using namespace modules;

    auto init_wrapper = [](segment::Builder& builder) {
        auto config     = nlohmann::json();
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
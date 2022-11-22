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

#pragma once

#include "../test_srf.hpp"  // IWYU pragma: keep

#include "srf/core/executor.hpp"  // IWYU pragma: keep
#include "srf/options/options.hpp"
#include "srf/options/topology.hpp"
#include "srf/pipeline/pipeline.hpp"
#include "srf/segment/builder.hpp"  // IWYU pragma: keep
#include "srf/segment/egress_ports.hpp"
#include "srf/segment/ingress_ports.hpp"
#include "srf/segment/segment.hpp"  // IWYU pragma: keep

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

// IWYU pragma: no_include "gtest/gtest_pred_impl.h"

namespace srf::segment {
struct ObjectProperties;
}

class TestSegmentResources
{
  public:
    TestSegmentResources() = default;

    static std::unique_ptr<Options> make_options()
    {
        auto options = std::make_unique<Options>();
        options->topology().user_cpuset("0");
        return options;
    }
};

class TestModules : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_pipeline  = pipeline::make_pipeline();
        m_resources = std::make_shared<TestSegmentResources>();
    }

    void TearDown() override {}

    std::unique_ptr<pipeline::Pipeline> m_pipeline;
    std::shared_ptr<TestSegmentResources> m_resources;
};

using TestModuleRegistry = TestModules;  // NOLINT
using TestModuleUtil     = TestModules;  // NOLINT
using TestSegmentModules = TestModules;  // NOLINT

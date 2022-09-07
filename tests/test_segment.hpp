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

#include "test_srf.hpp"  // IWYU pragma: keep

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

class SegmentTests : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_pipeline  = pipeline::make_pipeline();
        m_resources = std::make_shared<TestSegmentResources>();
    }

    void TearDown() override {}

    using ingress_types_t = segment::IngressPorts<int, int, double, std::string, float, std::size_t>;
    using egress_types_t  = segment::EgressPorts<float, double, unsigned int, float>;

    bool m_initializer_called = false;

    std::shared_ptr<segment::ObjectProperties> m_w;
    std::shared_ptr<segment::ObjectProperties> m_z;

    ingress_types_t m_ingress_multi_port = ingress_types_t(
        {"test_in_int1", "test_in_int2", "test_in_double", "test_in_string", "test_in_float", "test_in_sizet"});

    egress_types_t m_egress_multi_port =
        egress_types_t({"test_out_float", "test_out_double", "test_out_uint", "test_out_float2"});

    // Sum of nodes created by Ingress Types and Egress Types
    size_t m_InterfaceNodeCount;

    std::function<void(segment::Builder&)> m_initializer = [this](segment::Builder& s) {
        this->m_initializer_called = true;
    };
    std::unique_ptr<pipeline::Pipeline> m_pipeline;
    std::shared_ptr<TestSegmentResources> m_resources;
};

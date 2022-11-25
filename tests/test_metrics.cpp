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

#include "./test_mrc.hpp"  // IWYU pragma: associated

#include "mrc/metrics/counter.hpp"
#include "mrc/metrics/registry.hpp"

#include <gtest/gtest.h>  // for AssertionResult, SuiteApiResolver, TestInfo, EXPECT_TRUE, Message, TEST_F, Test, TestFactoryImpl, TestPartResult

#include <string>  // for allocator, operator==, basic_string, string

using namespace mrc;
using namespace metrics;

class TestMetrics : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_registry = std::make_shared<mrc::metrics::Registry>();
    }
    void TearDown() override {}

    std::shared_ptr<mrc::metrics::Registry> m_registry;
};

TEST_F(TestMetrics, ThroughputCounter)
{
    auto counter = m_registry->make_counter("mrc_throughput_counters", {{"name", "test_counter"}});

    counter.increment();
    counter.increment(42);

    auto report = m_registry->collect_throughput_counters();

    EXPECT_EQ(report.size(), 1);
    EXPECT_EQ(report[0].name, "test_counter");
    EXPECT_EQ(report[0].count, 43);
}

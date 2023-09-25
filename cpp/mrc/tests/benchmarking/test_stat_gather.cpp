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

#include "test_stat_gather.hpp"

#include "../test_segment.hpp"

#include "mrc/benchmarking/trace_statistics.hpp"
#include "mrc/benchmarking/util.hpp"
#include "mrc/options/options.hpp"
#include "mrc/pipeline/executor.hpp"

#include <nlohmann/json.hpp>

#include <map>

using namespace mrc::benchmarking;

void stat_check_helper(nlohmann::json metrics,
                       std::size_t ch_read,
                       std::size_t receive,
                       std::size_t ch_write,
                       std::size_t emit)
{
    EXPECT_EQ(metrics["component_channel_read_total"].get<std::size_t>(), ch_read);
    EXPECT_EQ(metrics["component_receive_total"].get<std::size_t>(), receive);
    EXPECT_EQ(metrics["component_channel_write_total"].get<std::size_t>(), ch_write);
    EXPECT_EQ(metrics["component_emissions_total"].get<std::size_t>(), emit);
}

std::string build_global_name(const std::string& segment_name, const std::string& component_name)
{
    return "/" + segment_name + "/" + component_name;
}

namespace mrc {

TEST_F(StatGatherTest, TestStatisticsOperatorGather)
{
    TraceStatistics::reset();
    TraceStatistics::trace_operators(true);

    Executor executor(std::move(m_resources->make_options()));
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    std::string seg_name                      = "segment_stats_test";
    std::set<std::string> required_components = {"src", "internal_1", "internal_2", "sink"};

    auto framework_stats_info = TraceStatistics::aggregate();
    auto& metrics             = framework_stats_info["aggregations"]["components"]["metrics"];
    for (const auto& component : required_components)
    {
        EXPECT_EQ(metrics.contains(build_global_name(seg_name, component)), true)
            << build_global_name(seg_name, component) << " not found";
    }

    stat_check_helper(metrics[build_global_name(seg_name, "src")], 0, 0, 0, m_iterations);
    stat_check_helper(metrics[build_global_name(seg_name, "internal_1")], 0, m_iterations, 0, m_iterations);
    stat_check_helper(metrics[build_global_name(seg_name, "internal_2")], 0, m_iterations, 0, m_iterations);
    stat_check_helper(metrics[build_global_name(seg_name, "sink")], 0, m_iterations, 0, 0);

    TraceStatistics::reset();
}

TEST_F(StatGatherTest, TestStatisticsChannelGather)
{
    TraceStatistics::reset();
    TraceStatistics::trace_channels(true);

    Executor executor(std::move(m_resources->make_options()));
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    std::string seg_name                      = "segment_stats_test";
    std::set<std::string> required_components = {"src", "internal_1", "internal_2", "sink"};

    auto framework_stats_info = TraceStatistics::aggregate();
    auto& metrics             = framework_stats_info["aggregations"]["components"]["metrics"];
    for (const auto& component : required_components)
    {
        EXPECT_EQ(metrics.contains(build_global_name(seg_name, component)), true)
            << build_global_name(seg_name, component) << " not found";
    }

    stat_check_helper(metrics[build_global_name(seg_name, "src")], 0, 0, m_iterations, 0);
    stat_check_helper(metrics[build_global_name(seg_name, "internal_1")], m_iterations, 0, m_iterations, 0);
    stat_check_helper(metrics[build_global_name(seg_name, "internal_2")], m_iterations, 0, m_iterations, 0);
    stat_check_helper(metrics[build_global_name(seg_name, "sink")], m_iterations, 0, 0, 0);

    TraceStatistics::reset();
}

TEST_F(StatGatherTest, TestStatisticsFullGather)
{
    TraceStatistics::reset();
    TraceStatistics::trace_channels(true);
    TraceStatistics::trace_operators(true);

    Executor executor(std::move(m_resources->make_options()));
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    std::string seg_name = "segment_stats_test";

    auto framework_stats_info = TraceStatistics::aggregate();
    auto& metrics             = framework_stats_info["aggregations"]["components"]["metrics"];
    for (const auto& component : m_components)
    {
        EXPECT_EQ(metrics.contains(build_global_name(seg_name, component)), true)
            << build_global_name(seg_name, component) << " not found";
    }

    stat_check_helper(metrics[build_global_name(seg_name, "src")], 0, 0, m_iterations, m_iterations);
    stat_check_helper(metrics[build_global_name(seg_name, "internal_1")],
                      m_iterations,
                      m_iterations,
                      m_iterations,
                      m_iterations);
    stat_check_helper(metrics[build_global_name(seg_name, "internal_2")],
                      m_iterations,
                      m_iterations,
                      m_iterations,
                      m_iterations);
    stat_check_helper(metrics[build_global_name(seg_name, "sink")], m_iterations, m_iterations, 0, 0);

    TraceStatistics::reset();
}

}  // namespace mrc

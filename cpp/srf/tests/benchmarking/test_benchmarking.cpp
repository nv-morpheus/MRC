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

#include "test_benchmarking.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

TEST_F(LatencyBenchmarkTests, BenchmarkingLatencyTracer)
{
    using nlohmann::json;

    m_watcher->tracer_count(m_iterations);
    m_watcher->reset();
    m_watcher->trace_until_notified();

    auto aggregation      = m_watcher->aggregate_tracers();
    const auto& json_data = aggregation->to_json();
    const auto& metadata  = json_data["metadata"];
    const auto& counters  = json_data["aggregations"]["metrics"]["counter"];

    EXPECT_EQ(metadata["node_count"], m_components.size());
    EXPECT_EQ(metadata["tracer_count"], m_iterations);

    std::string mean_latency_metric = "component_latency_seconds_mean";
    EXPECT_EQ(counters.contains(mean_latency_metric), true);
    const auto& mean_throughput_counters = counters[mean_latency_metric];
    for (const auto& counter : mean_throughput_counters)
    {
        // Very loose expectation. Basically that the component latency is less
        // than the overall average segment latency.
        EXPECT_LT(counter["value"],
                  metadata["tracer_count"].get<std::size_t>() / metadata["elapsed_time"].get<double>());
    }
}

TEST_F(ThroughputBenchmarkTests, BenchmarkingThroughputTracer)
{
    using nlohmann::json;

    m_watcher->tracer_count(m_iterations);
    m_watcher->reset();
    m_watcher->trace_until_notified();

    auto aggregation      = m_watcher->aggregate_tracers();
    const auto& json_data = aggregation->to_json();
    const auto& metadata  = json_data["metadata"];
    const auto& counters  = json_data["aggregations"]["metrics"]["counter"];

    EXPECT_EQ(metadata["node_count"], m_components.size());
    EXPECT_EQ(metadata["tracer_count"], m_iterations);

    std::string mean_throughput_metric = "component_mean_throughput";
    EXPECT_EQ(counters.contains(mean_throughput_metric), true);
    const auto& mean_throughput_counters = counters[mean_throughput_metric];
    for (const auto& counter : mean_throughput_counters)
    {
        // EXPECT_GT(counter["value"],
        //          metadata["tracer_count"].get<std::size_t>() / metadata["elapsed_time"].get<double>());
    }

    std::string process_total_metric_name = "component_processed_tracers_total";
    EXPECT_EQ(counters.contains(process_total_metric_name), true);
    const auto& processed_totals = counters[process_total_metric_name];
    for (const auto& counter : processed_totals)
    {
        EXPECT_EQ(counter["value"], m_iterations);
    }
}
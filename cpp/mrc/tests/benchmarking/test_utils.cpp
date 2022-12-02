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
#include "test_stat_gather.hpp"

#include "mrc/benchmarking/trace_statistics.hpp"
#include "mrc/benchmarking/util.hpp"

#include <nlohmann/json.hpp>
#include <prometheus/registry.h>

using namespace mrc::benchmarking;

TEST_F(StatGatherTest, TestPrometheusConversionForFrameworkStats)
{
    using nlohmann::json;
    TraceStatistics::reset();
    TraceStatistics::trace_channels(true);
    TraceStatistics::trace_operators(true);

    Executor executor(std::move(m_resources->make_options()));
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    auto framework_stats_info    = TraceStatistics::aggregate();
    auto& json_component_metrics = framework_stats_info["aggregations"]["components"]["metrics"];
    auto& json_metrics           = framework_stats_info["aggregations"]["metrics"]["counter"];

    auto prometheus_registry = mrc::benchmarking::json_to_prometheus(framework_stats_info);

    auto metrics = prometheus_registry->Collect();
    for (auto& metric_family : metrics)
    {
        if (metric_family.name == "segment_metadata")
        {
            continue;
        }
        // Ensure that every prometheus a metric_family is associated with each component.
        EXPECT_EQ(json_metrics.contains(metric_family.name), true);
        for (json::iterator component_it = json_component_metrics.begin(); component_it != json_component_metrics.end();
             ++component_it)
        {
            EXPECT_EQ(component_it.value().contains(metric_family.name), true);
        }

        for (auto& metric : metric_family.metric)
        {
            std::string component_id;
            for (auto& label : metric.label)
            {
                if (label.name == "component_id")
                {
                    component_id = label.value;
                    break;
                }
            }
            // Check that every component_id registered with prometheus corresponds to an expected segment
            // component (node).
            EXPECT_EQ(json_component_metrics.contains(component_id), true);

            // Spot checks to be sure basic metrics match what we expect. Not-exhaustive.
            // We expect sources to have no reads (because we're not looking at multi-segment) and
            // sinks to have no writes.
            if (component_id == "src")
            {
                if (metric_family.name == "component_channel_write_total")
                {
                    EXPECT_EQ(metric.counter.value, m_iterations);
                }
                if (metric_family.name == "component_channel_read_total")
                {
                    EXPECT_EQ(metric.counter.value, 0);
                }
            }
            else if (component_id == "sink")
            {
                if (metric_family.name == "component_channel_read_total")
                {
                    EXPECT_EQ(metric.counter.value, m_iterations);
                }
                if (metric_family.name == "component_channel_write_total")
                {
                    EXPECT_EQ(metric.counter.value, 0);
                }
            }
            else if (component_id == "internal_1" || component_id == "internal_2")
            {
                if (metric_family.name == "component_channel_read_total")
                {
                    EXPECT_EQ(metric.counter.value, m_iterations);
                }
                if (metric_family.name == "component_channel_write_total")
                {
                    EXPECT_EQ(metric.counter.value, m_iterations);
                }
            }
        }
    }

    TraceStatistics::reset();
}

TEST_F(LatencyBenchmarkTests, TestPrometheusConversionForWatcherTraces)
{
    using nlohmann::json;
    TraceStatistics::reset();
    TraceStatistics::trace_channels(true);
    TraceStatistics::trace_operators(true);

    m_watcher->tracer_count(m_iterations);
    m_watcher->reset();
    m_watcher->trace_until_notified();

    auto aggregation      = m_watcher->aggregate_tracers();
    const auto& json_data = aggregation->to_json();
    const auto& metadata  = json_data["metadata"];
    const auto& counters  = json_data["aggregations"]["metrics"]["counter"];

    auto prometheus_registry = mrc::benchmarking::json_to_prometheus(json_data);

    auto metrics = prometheus_registry->Collect();
    for (auto& metric_family : metrics)
    {
        if (metric_family.name == "segment_metadata")
        {
            continue;
        }
        // Ensure that every prometheus a metric_family is associated with each component.
        EXPECT_EQ(counters.contains(metric_family.name), true);

        for (auto& metric : metric_family.metric)
        {
            std::string component_id;
            for (auto& label : metric.label)
            {
                if (label.name == "component_id")
                {
                    component_id = label.value;
                    break;
                }
            }

            // Spot checks to be sure basic metrics match what we expect. Not-exhaustive.
            // We expect sources to have no reads (because we're not looking at multi-segment) and
            // sinks to have no writes.
            if (component_id == "src")
            {
                if (metric_family.name == "component_channel_write_total")
                {
                    EXPECT_EQ(metric.counter.value, m_iterations);
                }
                if (metric_family.name == "component_channel_read_total")
                {
                    EXPECT_EQ(metric.counter.value, 0);
                }
            }
            else if (component_id == "sink")
            {
                if (metric_family.name == "component_channel_read_total")
                {
                    EXPECT_EQ(metric.counter.value, m_iterations);
                }
                if (metric_family.name == "component_channel_write_total")
                {
                    EXPECT_EQ(metric.counter.value, 0);
                }
            }
            else if (component_id == "internal_1" || component_id == "internal_2")
            {
                if (metric_family.name == "component_channel_read_total")
                {
                    EXPECT_EQ(metric.counter.value, m_iterations);
                }
                if (metric_family.name == "component_channel_write_total")
                {
                    EXPECT_EQ(metric.counter.value, m_iterations);
                }
            }
        }
    }

    TraceStatistics::reset();
}

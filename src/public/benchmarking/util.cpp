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

#include "mrc/benchmarking/util.hpp"

#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <prometheus/counter.h>
#include <prometheus/family.h>
#include <prometheus/labels.h>
#include <prometheus/registry.h>

#include <chrono>
#include <map>
#include <string>

using namespace prometheus;
using namespace nlohmann;

namespace mrc::benchmarking {
std::size_t TimeUtil::s_mean_steady_clock_call_unit_delay{100};
TimeUtil::time_resolution_unit_t TimeUtil::s_minimum_resolution{10};

void TimeUtil::init()
{
    TimeUtil::s_minimum_resolution = TimeUtil::time_resolution_unit_t(10);
    TimeUtil::estimate_steady_clock_delay();
}

void TimeUtil::estimate_steady_clock_delay()
{
    std::size_t total_delay_units{0};
    for (auto i = 0; i < SteadyClockDelayEstimateIterations; i++)
    {
        auto start = get_current_time_point();
        auto end   = get_current_time_point();
        total_delay_units += TimeUtil::time_resolution_unit_t(end - start).count();
    }

    s_mean_steady_clock_call_unit_delay = total_delay_units / SteadyClockDelayEstimateIterations;
}

TimeUtil::time_pt_t TimeUtil::get_current_time_point()
{
    return clock_t::now();
}

TimeUtil::time_pt_t TimeUtil::get_delay_compensated_time_point()
{
    return get_current_time_point() + TimeUtil::time_resolution_unit_t(s_mean_steady_clock_call_unit_delay);
}

}  // namespace mrc::benchmarking

std::shared_ptr<Registry> mrc::benchmarking::json_to_prometheus(const json& json_data)
{
    using namespace mrc::benchmarking;

    auto registry = std::make_shared<Registry>();

    CHECK(json_data.contains("metadata"));
    const auto& metadata = json_data["metadata"];

    auto& meta_counter = prometheus::BuildCounter().Name("segment_metadata").Help("").Register(*registry);

    auto& trace_duration_counter = meta_counter.Add({{"type", "trace_duration"}});
    trace_duration_counter.Increment(metadata["elapsed_time"]);

    if (metadata.contains("tracer_count"))
    {
        auto& trace_tracers_counter = meta_counter.Add({{"type", "tracer_count"}});
        trace_tracers_counter.Increment(metadata["tracer_count"]);
    }

    if (metadata.contains("node_count"))
    {
        auto& trace_maxnodes_counter = meta_counter.Add({{"type", "node_count"}});
        trace_maxnodes_counter.Increment(metadata["node_count"]);
    }

    CHECK(json_data.contains("aggregations") && json_data["aggregations"].contains("metrics"));

    const auto& metrics = json_data["aggregations"]["metrics"];
    for (auto it = metrics.begin(); it != metrics.end(); it++)
    {
        if (it.key() == "counter")
        {
            json_counter_to_prometheus(it.value(), *registry);
        }
    }

    return registry;
}

void mrc::benchmarking::json_counter_to_prometheus(const json& counters, Registry& registry)
{
    for (auto it = counters.begin(); it != counters.end(); it++)
    {
        // Counters consists of key: value pairs with the counter's family name as the key, and a json::array of
        //  dimensional counter values.
        auto& counter_family = BuildCounter().Name(it.key()).Help("").Register(registry);

        // counter array
        const auto& counter_array = it.value();
        for (auto arr_it : counter_array)
        {
            CHECK(arr_it.contains("value"));

            prometheus::Labels metric_labels;
            if (arr_it.contains("labels"))
            {
                auto labels = arr_it["labels"];
                for (json::iterator label_it = labels.begin(); label_it != labels.end(); ++label_it)
                {
                    metric_labels[label_it.key()] = label_it.value().get<std::string>();
                }
            }

            auto& counter_instance = counter_family.Add(metric_labels);
            counter_instance.Increment(arr_it["value"]);
        }
    }
}

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

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <memory>

namespace prometheus {
class Registry;
}

namespace mrc::benchmarking {

struct TimeUtil
{
    using clock_t                = std::chrono::steady_clock;
    using time_pt_t              = std::chrono::time_point<clock_t>;
    using time_resolution_unit_t = std::chrono::nanoseconds;

    static constexpr std::size_t SteadyClockDelayEstimateIterations = 1e6;
    static constexpr double NsToSec                                 = 1 / 1e9;
    static time_resolution_unit_t s_minimum_resolution;
    static std::size_t s_mean_steady_clock_call_unit_delay;

    /**
     * @brief initialize static members.
     */
    static void init();

    /**
     * @brief Compute average call time for steady_clock::now()
     */
    static void estimate_steady_clock_delay();

    /**
     * @brief Get the current time point
     * @return Return the current time point represented by TimeUtil::time_pt_t
     */
    static time_pt_t get_current_time_point();

    /**
     * @brief Get the current time point, adjusting the returned value forward by the expected mean call time of
     * clock_t::now().
     * @return Time point representing the expected time when the call to clock_t::now() returns.
     */
    static time_pt_t get_delay_compensated_time_point();
};

/**
 * TODO(Devin): Enforce schema
 * @brief Conversion routine taking a set of benchmarking data in JSON format and transforming it into a Prometheus
 * Registry.
 * @param json_data
 * @return shared pointer to a newly created prometheus registry.
 */
std::shared_ptr<prometheus::Registry> json_to_prometheus(const nlohmann::json&);

/**
 * @brief Conversion routine to translate a set of JSON counter metrics into Prometheus counters.
 * @param counters JSON object containing an array of counter metrics
 */
void json_counter_to_prometheus(const nlohmann::json&, prometheus::Registry&);
}  // namespace mrc::benchmarking

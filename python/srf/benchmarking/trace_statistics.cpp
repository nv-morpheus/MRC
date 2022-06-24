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

#include "srf/benchmarking/trace_statistics.hpp"

#include "pysrf/types.hpp"  // IWYU pragma: keep
#include "pysrf/utils.hpp"

#include <nlohmann/json.hpp>  // IWYU pragma: keep
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>  // IYWU pragma: keep
#include <pybind11/pytypes.h>

namespace srf::pysrf {

namespace py = pybind11;

void init_tracer_stats_api(py::module_& m)
{
    m.def("get_tracing_stats", []() {
        nlohmann::json stats_data = srf::benchmarking::TraceStatistics::aggregate();

        return pysrf::cast_from_json(stats_data);
    });
    m.def("trace_operators",
          static_cast<void (*)(bool, bool)>(&srf::benchmarking::TraceStatistics::trace_operators),
          py::arg("trace"),
          py::arg("sync_immediate") = bool(true));
    m.def("trace_operators",
          static_cast<std::tuple<bool, bool> (*)()>(&srf::benchmarking::TraceStatistics::trace_operators));
    m.def("trace_channels",
          static_cast<void (*)(bool, bool)>(&srf::benchmarking::TraceStatistics::trace_channels),
          py::arg("trace"),
          py::arg("sync_immediate") = bool(true));
    m.def("trace_channels",
          static_cast<std::tuple<bool, bool> (*)()>(&srf::benchmarking::TraceStatistics::trace_channels));
    m.def("reset_tracing_stats", &srf::benchmarking::TraceStatistics::reset);
    m.def("sync_tracing_state", &srf::benchmarking::TraceStatistics::sync_state);
}
}  // namespace srf::pysrf

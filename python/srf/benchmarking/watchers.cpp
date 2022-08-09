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

#include "pysrf/watchers.hpp"

#include "pysrf/executor.hpp"  // IWYU pragma: keep
#include "pysrf/segment.hpp"

#include "srf/segment/builder.hpp"  // IWYU pragma: keep

#include <pybind11/attr.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include <pybind11/detail/descr.h>

namespace srf::pysrf {

namespace py = pybind11;

PYBIND11_MODULE(watchers, m)
{
    m.doc() = R"pbdoc()pbdoc";

    // Segment watcher allows for each tracer object to have a data payload. To simplify, for now, we'll assume
    // that the payload is a py::object.
    // auto SegmentWatcher = py::class_<srf::SegmentWatcher<py::object>>(m, "SegmentWatcher");
    auto PyLatencyWatcher = py::class_<pysrf::LatencyWatcher>(m, "LatencyWatcher");
    PyLatencyWatcher.def(py::init<std::shared_ptr<pysrf::Executor>>());
    PyLatencyWatcher.def(py::init<std::shared_ptr<pysrf::Executor>, std::function<void(pysrf::latency_ensemble_t&)>>());
    PyLatencyWatcher.def("aggregate_tracers", &pysrf::LatencyWatcher::aggregate_tracers_as_pydict);
    // PyLatencyWatcher.def("make_tracer_source", &pysrf::LatencyWatcher::create_rx_tracer_source<false>);
    PyLatencyWatcher.def("is_running", &pysrf::LatencyWatcher::is_running);
    PyLatencyWatcher.def("make_segment", pysrf::wrap_segment_init_callback(&pysrf::LatencyWatcher::make_segment));
    PyLatencyWatcher.def(
        "make_tracer_source", &pysrf::LatencyWatcher::make_tracer_source, py::return_value_policy::reference_internal);
    PyLatencyWatcher.def(
        "make_traced_node", &pysrf::LatencyWatcher::make_traced_node, py::return_value_policy::reference_internal);
    PyLatencyWatcher.def(
        "make_tracer_sink", &pysrf::LatencyWatcher::make_tracer_sink, py::return_value_policy::reference_internal);
    PyLatencyWatcher.def("reset", &pysrf::LatencyWatcher::reset, py::call_guard<py::gil_scoped_release>());
    PyLatencyWatcher.def("run", &pysrf::LatencyWatcher::run, py::call_guard<py::gil_scoped_release>());
    PyLatencyWatcher.def("shutdown", &pysrf::LatencyWatcher::shutdown, py::call_guard<py::gil_scoped_release>());
    PyLatencyWatcher.def("start_trace", &pysrf::LatencyWatcher::start_trace, py::call_guard<py::gil_scoped_release>());
    PyLatencyWatcher.def("stop_trace", &pysrf::LatencyWatcher::stop_trace, py::call_guard<py::gil_scoped_release>());
    PyLatencyWatcher.def(
        "trace_until_notified", &pysrf::LatencyWatcher::trace_until_notified, py::call_guard<py::gil_scoped_release>());
    PyLatencyWatcher.def("tracer_count", py::overload_cast<std::size_t>(&pysrf::LatencyWatcher::tracer_count));
    PyLatencyWatcher.def("tracing", &pysrf::LatencyWatcher::tracing);

    /** Throughput Watcher Begin **/
    auto PyThroughputWatcher = py::class_<pysrf::ThroughputWatcher>(m, "ThroughputWatcher");

    PyThroughputWatcher.def(py::init<std::shared_ptr<pysrf::Executor>>());
    PyThroughputWatcher.def(
        py::init<std::shared_ptr<pysrf::Executor>, std::function<void(pysrf::throughput_ensemble_t&)>>());
    PyThroughputWatcher.def("aggregate_tracers", &pysrf::ThroughputWatcher::aggregate_tracers_as_pydict);
    // PyThroughputWatcher.def("make_tracer_source", &pysrf::ThroughputWatcher::create_rx_tracer_source<false>);
    PyThroughputWatcher.def("is_running", &pysrf::ThroughputWatcher::is_running);
    PyThroughputWatcher.def("make_segment", pysrf::wrap_segment_init_callback(&pysrf::ThroughputWatcher::make_segment));
    PyThroughputWatcher.def("make_tracer_source",
                            &pysrf::ThroughputWatcher::make_tracer_source,
                            py::return_value_policy::reference_internal);
    PyThroughputWatcher.def(
        "make_traced_node", &pysrf::ThroughputWatcher::make_traced_node, py::return_value_policy::reference_internal);
    PyThroughputWatcher.def(
        "make_tracer_sink", &pysrf::ThroughputWatcher::make_tracer_sink, py::return_value_policy::reference_internal);
    PyThroughputWatcher.def("reset", &pysrf::ThroughputWatcher::reset, py::call_guard<py::gil_scoped_release>());
    PyThroughputWatcher.def("run", &pysrf::ThroughputWatcher::run, py::call_guard<py::gil_scoped_release>());
    PyThroughputWatcher.def("shutdown", &pysrf::ThroughputWatcher::shutdown, py::call_guard<py::gil_scoped_release>());
    PyThroughputWatcher.def(
        "start_trace", &pysrf::ThroughputWatcher::start_trace, py::call_guard<py::gil_scoped_release>());
    PyThroughputWatcher.def(
        "stop_trace", &pysrf::ThroughputWatcher::stop_trace, py::call_guard<py::gil_scoped_release>());
    PyThroughputWatcher.def("trace_until_notified",
                            &pysrf::ThroughputWatcher::trace_until_notified,
                            py::call_guard<py::gil_scoped_release>());
    PyThroughputWatcher.def("tracer_count", py::overload_cast<std::size_t>(&pysrf::ThroughputWatcher::tracer_count));
    PyThroughputWatcher.def("tracing", &pysrf::ThroughputWatcher::tracing);
}
}  // namespace srf::pysrf

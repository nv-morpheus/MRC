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

#include "pymrc/watchers.hpp"

#include "pymrc/executor.hpp"  // IWYU pragma: keep
#include "pymrc/segment.hpp"

#include "mrc/segment/builder.hpp"  // IWYU pragma: keep

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

namespace mrc::pymrc {

namespace py = pybind11;

PYBIND11_MODULE(watchers, m)
{
    m.doc() = R"pbdoc()pbdoc";

    // Segment watcher allows for each tracer object to have a data payload. To simplify, for now, we'll assume
    // that the payload is a py::object.
    // auto SegmentWatcher = py::class_<mrc::SegmentWatcher<py::object>>(m, "SegmentWatcher");
    auto PyLatencyWatcher = py::class_<pymrc::LatencyWatcher>(m, "LatencyWatcher");
    PyLatencyWatcher.def(py::init<std::shared_ptr<pymrc::Executor>>());
    PyLatencyWatcher.def(py::init<std::shared_ptr<pymrc::Executor>, std::function<void(pymrc::latency_ensemble_t&)>>());
    PyLatencyWatcher.def("aggregate_tracers", &pymrc::LatencyWatcher::aggregate_tracers_as_pydict);
    // PyLatencyWatcher.def("make_tracer_source", &pymrc::LatencyWatcher::create_rx_tracer_source<false>);
    PyLatencyWatcher.def("is_running", &pymrc::LatencyWatcher::is_running);
    PyLatencyWatcher.def("make_segment", pymrc::wrap_segment_init_callback(&pymrc::LatencyWatcher::make_segment));
    PyLatencyWatcher.def(
        "make_tracer_source", &pymrc::LatencyWatcher::make_tracer_source, py::return_value_policy::reference_internal);
    PyLatencyWatcher.def(
        "make_traced_node", &pymrc::LatencyWatcher::make_traced_node, py::return_value_policy::reference_internal);
    PyLatencyWatcher.def(
        "make_tracer_sink", &pymrc::LatencyWatcher::make_tracer_sink, py::return_value_policy::reference_internal);
    PyLatencyWatcher.def("reset", &pymrc::LatencyWatcher::reset, py::call_guard<py::gil_scoped_release>());
    PyLatencyWatcher.def("run", &pymrc::LatencyWatcher::run, py::call_guard<py::gil_scoped_release>());
    PyLatencyWatcher.def("shutdown", &pymrc::LatencyWatcher::shutdown, py::call_guard<py::gil_scoped_release>());
    PyLatencyWatcher.def("start_trace", &pymrc::LatencyWatcher::start_trace, py::call_guard<py::gil_scoped_release>());
    PyLatencyWatcher.def("stop_trace", &pymrc::LatencyWatcher::stop_trace, py::call_guard<py::gil_scoped_release>());
    PyLatencyWatcher.def(
        "trace_until_notified", &pymrc::LatencyWatcher::trace_until_notified, py::call_guard<py::gil_scoped_release>());
    PyLatencyWatcher.def("tracer_count", py::overload_cast<std::size_t>(&pymrc::LatencyWatcher::tracer_count));
    PyLatencyWatcher.def("tracing", &pymrc::LatencyWatcher::tracing);

    /** Throughput Watcher Begin **/
    auto PyThroughputWatcher = py::class_<pymrc::ThroughputWatcher>(m, "ThroughputWatcher");

    PyThroughputWatcher.def(py::init<std::shared_ptr<pymrc::Executor>>());
    PyThroughputWatcher.def(
        py::init<std::shared_ptr<pymrc::Executor>, std::function<void(pymrc::throughput_ensemble_t&)>>());
    PyThroughputWatcher.def("aggregate_tracers", &pymrc::ThroughputWatcher::aggregate_tracers_as_pydict);
    // PyThroughputWatcher.def("make_tracer_source", &pymrc::ThroughputWatcher::create_rx_tracer_source<false>);
    PyThroughputWatcher.def("is_running", &pymrc::ThroughputWatcher::is_running);
    PyThroughputWatcher.def("make_segment", pymrc::wrap_segment_init_callback(&pymrc::ThroughputWatcher::make_segment));
    PyThroughputWatcher.def("make_tracer_source",
                            &pymrc::ThroughputWatcher::make_tracer_source,
                            py::return_value_policy::reference_internal);
    PyThroughputWatcher.def(
        "make_traced_node", &pymrc::ThroughputWatcher::make_traced_node, py::return_value_policy::reference_internal);
    PyThroughputWatcher.def(
        "make_tracer_sink", &pymrc::ThroughputWatcher::make_tracer_sink, py::return_value_policy::reference_internal);
    PyThroughputWatcher.def("reset", &pymrc::ThroughputWatcher::reset, py::call_guard<py::gil_scoped_release>());
    PyThroughputWatcher.def("run", &pymrc::ThroughputWatcher::run, py::call_guard<py::gil_scoped_release>());
    PyThroughputWatcher.def("shutdown", &pymrc::ThroughputWatcher::shutdown, py::call_guard<py::gil_scoped_release>());
    PyThroughputWatcher.def(
        "start_trace", &pymrc::ThroughputWatcher::start_trace, py::call_guard<py::gil_scoped_release>());
    PyThroughputWatcher.def(
        "stop_trace", &pymrc::ThroughputWatcher::stop_trace, py::call_guard<py::gil_scoped_release>());
    PyThroughputWatcher.def("trace_until_notified",
                            &pymrc::ThroughputWatcher::trace_until_notified,
                            py::call_guard<py::gil_scoped_release>());
    PyThroughputWatcher.def("tracer_count", py::overload_cast<std::size_t>(&pymrc::ThroughputWatcher::tracer_count));
    PyThroughputWatcher.def("tracing", &pymrc::ThroughputWatcher::tracing);
}
}  // namespace mrc::pymrc

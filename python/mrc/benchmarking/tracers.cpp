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

#include "pymrc/types.hpp"  // IWYU pragma: keep

#include "mrc/benchmarking/tracer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <cstddef>  // for size_t
#include <memory>

namespace mrc::pymrc {

namespace py = pybind11;

void init_tracer_stats_api(py::module_& m);
void init_tracer_api(py::module_& m);

using latency_tracer_t    = mrc::benchmarking::TracerEnsemble<py::object, mrc::benchmarking::LatencyTracer>;
using throughput_tracer_t = mrc::benchmarking::TracerEnsemble<py::object, mrc::benchmarking::ThroughputTracer>;

// TODO (Devin): Not supporting direct tracers yet, file still needs to be implemented.
PYBIND11_MODULE(tracers, m)
{
    m.doc() = R"pbdoc()pbdoc";

    /**
     * @brief define tracer stats module components
     */
    init_tracer_stats_api(m);

    /**
     * @brief define tracer implementations for use with segment watchers
     */
    // pymrc::init_tracer_api(m);

    /**
     * @brief Tracer objects are packaged into tracer ensembles; we'll just support tracer's with py::object payloads
     *  for now.
     */
    auto LatencyTracer = py::class_<latency_tracer_t, std::shared_ptr<latency_tracer_t>>(m, "LatencyTracer");
    LatencyTracer.def(py::init<std::size_t>());

    // TODO(devin)
    // LatencyTracer.def("add_counters", &mrc::LatencyTracer::add_counters);
    LatencyTracer.def_static("aggregate", [](py::object& obj_type, py::list ensemble_tracers) {
        // Something is broken with calling static members
    });
    LatencyTracer.def("emit", &latency_tracer_t::emit);

    /**
     * @brief ThroughputTracer
     */
    auto ThroughputTracer =
        py::class_<throughput_tracer_t, std::shared_ptr<throughput_tracer_t>>(m, "ThroughputTracer");
    ThroughputTracer.def(py::init<std::size_t>());
    // ThroughputTracer.def("add_counters", &ThroughputTracerT::add_counters);
    // ThroughputTracer.def("aggregate", &ThroughputTracerT::aggregate);
}
}  // namespace mrc::pymrc

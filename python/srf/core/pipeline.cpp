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

#include "pysrf/pipeline.hpp"

#include "pysrf/segment.hpp"
#include "pysrf/utils.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

// IWYU thinks we need array for py::class_<Pipeline>
// IWYU pragma: no_include <array>

namespace srf::pysrf {

namespace py = pybind11;

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(pipeline, m)
{
    m.doc() = R"pbdoc(
        Python bindings for SRF pipelines
        -------------------------------
        .. currentmodule:: pipeline
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // Common must be first in every module
    pysrf::import(m, "srf.core.common");
    pysrf::import(m, "srf.core.segment");

    m.attr("SRF_MAX_INGRESS_PORTS") = SRF_MAX_INGRESS_PORTS;
    m.attr("SRF_MAX_EGRESS_PORTS")  = SRF_MAX_EGRESS_PORTS;
    py::class_<Pipeline>(m, "Pipeline")
        .def(py::init<>())
        .def(
            "make_segment",
            wrap_segment_init_callback(
                static_cast<void (Pipeline::*)(const std::string&, const std::function<void(srf::segment::Builder&)>&)>(
                    &Pipeline::make_segment)))
        .def("make_segment",
             wrap_segment_init_callback(
                 static_cast<void (Pipeline::*)(const std::string&,
                                                const std::vector<std::string>&,
                                                const std::vector<std::string>&,
                                                const std::function<void(srf::segment::Builder&)>&)>(
                     &Pipeline::make_segment)));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace srf::pysrf

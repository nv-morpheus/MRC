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

#include "srf/segment/builder.hpp"  // IWYU pragma: keep
#include "srf/version.hpp"

#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include <pybind11/detail/descr.h>
// IWYU thinks we need array for py::class_<Pipeline>
// IWYU pragma: no_include <array>

namespace srf::pysrf {

namespace py = pybind11;

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(pipeline, module)
{
    module.doc() = R"pbdoc(
        Python bindings for SRF pipelines
        -------------------------------
        .. currentmodule:: pipeline
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // Common must be first in every module
    pysrf::import(module, "srf.core.common");
    pysrf::import(module, "srf.core.segment");

    py::class_<Pipeline>(module, "Pipeline")
        .def(py::init<>())
        .def(
            "make_segment",
            wrap_segment_init_callback(
                static_cast<void (Pipeline::*)(const std::string&, const std::function<void(srf::segment::Builder&)>&)>(
                    &Pipeline::make_segment)))
        .def("make_segment",
             wrap_segment_init_callback(
                 static_cast<void (Pipeline::*)(
                     const std::string&, py::list, py::list, const std::function<void(srf::segment::Builder&)>&)>(
                     &Pipeline::make_segment)));

    std::stringstream sstream;
    sstream << srf_VERSION_MAJOR << "." << srf_VERSION_MINOR << "." << srf_VERSION_PATCH;

    module.attr("__version__") = sstream.str();
}
}  // namespace srf::pysrf

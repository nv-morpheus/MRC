/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/pipeline.hpp"

#include "pymrc/segment.hpp"
#include "pymrc/utils.hpp"

#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <sstream>

namespace mrc::pymrc {

namespace py = pybind11;

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(pipeline, py_mod)
{
    py_mod.doc() = R"pbdoc(
        Python bindings for MRC pipelines
        -------------------------------
        .. currentmodule:: pipeline
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // Common must be first in every module
    pymrc::import(py_mod, "mrc.core.common");
    pymrc::import(py_mod, "mrc.core.segment");

    py::class_<Pipeline>(py_mod, "Pipeline")
        .def(py::init<>())
        .def(
            "make_segment",
            wrap_segment_init_callback(
                static_cast<void (Pipeline::*)(const std::string&, const std::function<void(mrc::segment::IBuilder&)>&)>(
                    &Pipeline::make_segment)))
        .def("make_segment",
             wrap_segment_init_callback(
                 static_cast<void (Pipeline::*)(const std::string&,
                                                py::list,
                                                py::list,
                                                const std::function<void(mrc::segment::IBuilder&)>&)>(
                     &Pipeline::make_segment)));

    py_mod.attr("__version__") = MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "."
                                                                  << mrc_VERSION_PATCH);
}
}  // namespace mrc::pymrc

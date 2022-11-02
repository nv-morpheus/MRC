/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pysrf/logging.hpp"

#include "srf/core/logging.hpp"
#include "srf/version.hpp"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <sstream>

namespace srf::pysrf {

namespace py = pybind11;
using namespace std::string_literals;

PYBIND11_MODULE(logging, module)
{
    module.doc() = R"pbdoc(
        Python bindings for SRF logging
        -------------------------------
        .. currentmodule:: logging
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    module.def(
        "init_logging",
        &init_logging,
        "Initializes SRF's logger, The return value inidicates if the logger was initialized, which will be `True` "
        "on the first call, and `False` for all subsequant calls.",
        py::arg("logname"),
        py::arg("py_level") = py_log_levels::INFO);

    module.def("is_initialized", &srf::is_initialized, "Checks if SRF's logger has been initialized.");

    module.def("get_level", &get_level, "Gets the log level for SRF's logger.");

    module.def("set_level", &set_level, "Sets the log level for SRF's logger.", py::arg("py_level"));

    module.def("log",
               &log,
               "Logs a message to SRF's logger.",
               py::arg("msg"),
               py::arg("py_level") = py_log_levels::INFO,
               py::arg("filename") = ""s,
               py::arg("line")     = 0);

    std::stringstream sstream;
    sstream << srf_VERSION_MAJOR << "." << srf_VERSION_MINOR << "." << srf_VERSION_PATCH;

    module.attr("__version__") = sstream.str();
}
}  // namespace srf::pysrf

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

#include <pysrf/logging.hpp>

#include <srf/core/logging.hpp>

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

namespace srf::pysrf {

namespace py = pybind11;
using namespace std::string_literals;

PYBIND11_MODULE(logging, m)
{
    m.doc() = R"pbdoc(
        Python bindings for SRF logging
        -------------------------------
        .. currentmodule:: logging
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    m.def("init_logging",
          &init_logging,
          "Initializes Srf's logger, The return value inidicates if the logger was initialized, which will be `True` "
          "on the first call, and `False` for all subsequant calls.",
          py::arg("logname"),
          py::arg("py_level") = py_log_levels::INFO);

    m.def("is_initialized", &srf::is_initialized, "Checks if Srf's logger has been initialized.");

    m.def("get_level", &get_level, "Gets the log level for Srf's logger.");

    m.def("set_level", &set_level, "Sets the log level for Srf's logger.", py::arg("py_level"));

    m.def("log",
          &log,
          "Logs a message to Srf's logger.",
          py::arg("msg"),
          py::arg("py_level") = py_log_levels::INFO,
          py::arg("filename") = ""s,
          py::arg("line")     = 0);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace srf::pysrf

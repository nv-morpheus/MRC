/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/logging.hpp"

#include "mrc/core/logging.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <sstream>

namespace mrc::pymrc {

namespace py = pybind11;
using namespace std::string_literals;

PYBIND11_MODULE(logging, py_mod)
{
    py_mod.doc() = R"pbdoc(
        Python bindings for MRC logging
        -------------------------------
        .. currentmodule:: logging
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    py_mod.def("init_logging",
               &init_logging,
               "Initializes MRC's logger, The return value inidicates if the logger was initialized, which will be "
               "`True` "
               "on the first call, and `False` for all subsequant calls.",
               py::arg("logname"),
               py::arg("py_level") = py_log_levels::INFO);

    py_mod.def("is_initialized", &mrc::is_initialized, "Checks if MRC's logger has been initialized.");

    py_mod.def("get_level", &get_level, "Gets the log level for MRC's logger.");

    py_mod.def("set_level", &set_level, "Sets the log level for MRC's logger.", py::arg("py_level"));

    py_mod.def("log",
               &log,
               "Logs a message to MRC's logger.",
               py::arg("msg"),
               py::arg("py_level") = py_log_levels::INFO,
               py::arg("filename") = ""s,
               py::arg("line")     = 0);

    py_mod.attr("__version__") = MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "."
                                                                  << mrc_VERSION_PATCH);
}
}  // namespace mrc::pymrc

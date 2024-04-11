/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/utils.hpp"

#include "pymrc/utilities/json_values.hpp"  // for JSONValues

#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <pybind11/cast.h>
#include <pybind11/gil.h>  // for gil_scoped_acquire
#include <pybind11/pybind11.h>

#include <sstream>
#include <stdexcept>

namespace mrc::pytests {

namespace py = pybind11;

// Simple test class which uses pybind11's `gil_scoped_acquire` class in the destructor. Needed to repro #362
struct RequireGilInDestructor
{
    ~RequireGilInDestructor()
    {
        // Grab the GIL
        py::gil_scoped_acquire gil;
    }
};

pymrc::JSONValues roundtrip_cast(pymrc::JSONValues v)
{
    return v;
}

PYBIND11_MODULE(utils, py_mod)
{
    py_mod.doc() = R"pbdoc()pbdoc";

    pymrc::import(py_mod, "mrc");

    py_mod.def(
        "throw_cpp_error",
        [](std::string msg = "") {
            if (msg.empty())
            {
                msg = "Exception from C++ code";
            }

            throw std::runtime_error(msg);
        },
        py::arg("msg") = "");

    py::class_<RequireGilInDestructor>(py_mod, "RequireGilInDestructor").def(py::init<>());

    py_mod.def("roundtrip_cast", &roundtrip_cast, py::arg("v"));

    py_mod.attr("__version__") = MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "."
                                                                  << mrc_VERSION_PATCH);
}
}  // namespace mrc::pytests

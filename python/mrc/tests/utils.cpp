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

#include "pymrc/utils.hpp"

#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

#include <iostream>  // DO NOT MERGE
#include <memory>
#include <sstream>
#include <stdexcept>

namespace mrc::pytests {

namespace py = pybind11;

// Simple test class which acquires the GIL in it's destructor
struct ObjUsingGil
{
    ObjUsingGil() = default;
    ~ObjUsingGil()
    {
        std::cerr << "ObjUsingGil::~ObjUsingGil()" << std::endl << std::flush;
        py::gil_scoped_acquire gil;
        std::cerr << "ObjUsingGil::~ObjUsingGil()+gil" << std::endl << std::flush;
    }
};

struct ObjCallingGC
{
    ObjCallingGC() = default;

    static void finalize()
    {
        std::cerr << "ObjCallingGC::finalize()" << std::endl << std::flush;
        py::gil_scoped_acquire gil;
        py::object gc = py::module::import("gc");
        std::cerr << "ObjCallingGC::finalize() - calling collect" << std::endl << std::flush;
        gc.attr("collect")();
    }
};

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

    py::class_<ObjUsingGil>(py_mod, "ObjUsingGil").def(py::init<>());

    py::class_<ObjCallingGC>(py_mod, "ObjCallingGC").def(py::init<>()).def_static("finalize", &ObjCallingGC::finalize);

    py_mod.attr("__version__") = MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "."
                                                                  << mrc_VERSION_PATCH);
}
}  // namespace mrc::pytests

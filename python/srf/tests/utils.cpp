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

#include "pysrf/utils.hpp"

#include "srf/utils/string_utils.hpp"
#include "srf/version.hpp"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

#include <ostream>
#include <stdexcept>

namespace srf::pytests {

namespace py = pybind11;

PYBIND11_MODULE(utils, module)
{
    module.doc() = R"pbdoc()pbdoc";

    pysrf::import(module, "srf");

    module.def(
        "throw_cpp_error",
        [](std::string msg = "") {
            if (msg.empty())
            {
                msg = "Exception from C++ code";
            }

            throw std::runtime_error(msg);
        },
        py::arg("msg") = "");

    module.attr("__version__") =
        SRF_CONCAT_STR(srf_VERSION_MAJOR << "." << srf_VERSION_MINOR << "." << srf_VERSION_PATCH);
}
}  // namespace srf::pytests

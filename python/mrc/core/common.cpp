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

#include "pymrc/port_builders.hpp"
#include "pymrc/types.hpp"

#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rxcpp/rx.hpp>

#include <memory>
#include <sstream>

namespace mrc::pymrc {

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(common, py_mod)
{
    py_mod.doc() = R"pbdoc(
        Python bindings for MRC common functionality / utilities
        -------------------------------
        .. currentmodule:: common
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    PortBuilderUtil::register_port_util<PyHolder>();

    py_mod.attr("__version__") = MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "."
                                                                  << mrc_VERSION_PATCH);
}
}  // namespace mrc::pymrc

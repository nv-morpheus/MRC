/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/operators.hpp"

#include "pymrc/utilities/function_wrappers.hpp"  // IWYU pragma: keep
#include "pymrc/utils.hpp"

#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <sstream>

namespace mrc::pymrc {

namespace py = pybind11;

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(operators, py_mod)
{
    py_mod.doc() = R"pbdoc(
        Python bindings for MRC operators
        -------------------------------
        .. currentmodule:: operators
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // Common must be first in every module
    pymrc::import(py_mod, "mrc.core.common");
    pymrc::import(py_mod, "mrc.core.subscriber");

    py::class_<PythonOperator>(py_mod, "Operator").def_property_readonly("name", &OperatorProxy::get_name);

    py_mod.def("build", &OperatorsProxy::build);
    py_mod.def("filter", &OperatorsProxy::filter);
    py_mod.def("flatten", &OperatorsProxy::flatten);
    py_mod.def("map", &OperatorsProxy::map);
    py_mod.def("on_completed", &OperatorsProxy::on_completed);
    py_mod.def("pairwise", &OperatorsProxy::pairwise);
    py_mod.def("to_list", &OperatorsProxy::to_list);

    py_mod.attr("__version__") = MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "."
                                                                  << mrc_VERSION_PATCH);
}
}  // namespace mrc::pymrc

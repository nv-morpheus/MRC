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

#include <pysrf/operators.hpp>

#include <pysrf/utils.hpp>

#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/pybind11.h>

#include <array>

namespace srf::pysrf {

namespace py = pybind11;

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(operators, m)
{
    m.doc() = R"pbdoc(
        Python bindings for SRF operators
        -------------------------------
        .. currentmodule:: operators
        .. autosummary::
           :toctree: _generate
    )pbdoc";


    // Common must be first in every module
    pysrf::import(m, "srf.core.common");

    py::class_<PythonOperator>(m, "Operator").def_property_readonly("name", &OperatorProxy::get_name);

    m.def("filter", &OperatorsProxy::filter);
    m.def("flatten", &OperatorsProxy::flatten);
    m.def("map", &OperatorsProxy::map);
    m.def("on_completed", &OperatorsProxy::on_completed);
    m.def("pairwise", &OperatorsProxy::pairwise);
    m.def("to_list", &OperatorsProxy::to_list);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace srf::pysrf

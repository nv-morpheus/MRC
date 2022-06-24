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

#include "pysrf/types.hpp"
#include "pysrf/utils.hpp"

#include "srf/runnable/launch_options.hpp"
#include "srf/segment/object.hpp"

#include <pybind11/pybind11.h>  // IWYU pragma: keep

#include <cstddef>
#include <memory>
#include <string>

namespace srf::pysrf {
namespace py = pybind11;

PYBIND11_MODULE(node, m)
{
    m.doc() = R"pbdoc()pbdoc";

    // Common must be first in every module
    pysrf::import(m, "srf.core.common");

    py::class_<srf::runnable::LaunchOptions>(m, "LaunchOptions")
        .def_readwrite("pe_count", &srf::runnable::LaunchOptions::pe_count)
        .def_readwrite("engines_per_pe", &srf::runnable::LaunchOptions::engines_per_pe)
        .def_readwrite("engine_factory_name", &srf::runnable::LaunchOptions::engine_factory_name);

    py::class_<srf::segment::ObjectProperties, std::shared_ptr<srf::segment::ObjectProperties>>(m, "SegmentObject")
        .def_property_readonly("name", &PyNode::name)
        .def_property_readonly("launch_options",
                               py::overload_cast<>(&srf::segment::ObjectProperties::launch_options),
                               py::return_value_policy::reference_internal);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace srf::pysrf

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

#include "pysrf/segment_modules.hpp"

#include "pysrf/module_registry.hpp"
#include "pysrf/utils.hpp"

#include "srf/experimental/modules/segment_modules.hpp"
#include "srf/segment/builder.hpp"

#include <pybind11/pybind11.h>

#include <memory>

// IWYU thinks the Segment.def calls need array and vector
// IWYU pragma: no_include <array>
// IWYU pragma: no_include <vector>
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include <pybind11/detail/descr.h>

namespace srf::pysrf {

namespace py = pybind11;

PYBIND11_MODULE(segment_modules, m)
{
    m.doc() = R"pbdoc(
       Python bindings for SRF Segment Modules
       -------------------------------
       .. currentmodule:: Segment Modules
       .. autosummary::
          :toctree: _generate
   )pbdoc";

    pysrf::import(m, "srf.core.common");
    pysrf::import_module_object(m, "srf.core.segment", "Builder");

    auto SegmentModule =
        py::class_<srf::modules::SegmentModule, std::shared_ptr<srf::modules::SegmentModule>>(m, "SegmentModule");
    auto SegmentModuleRegistry = py::class_<ModuleRegistryProxy>(m, "ModuleRegistry");

    /** Segment Module Interface Declarations **/
    // TODO(bhargav): SegmentModule constructor binding

    SegmentModule.def("config", &SegmentModuleProxy::config);

    SegmentModule.def("component_prefix", &SegmentModuleProxy::component_prefix);

    SegmentModule.def("input_port", &SegmentModuleProxy::input_port, py::arg("input_id"));

    SegmentModule.def("input_ports", &SegmentModuleProxy::input_ports);

    SegmentModule.def("module_name", &SegmentModuleProxy::module_name);

    SegmentModule.def("name", &SegmentModuleProxy::name);

    SegmentModule.def("output_port", &SegmentModuleProxy::output_port, py::arg("output_id"));

    SegmentModule.def("output_ports", &SegmentModuleProxy::output_ports);

    SegmentModule.def("input_ids", &SegmentModuleProxy::input_ids);

    SegmentModule.def("output_ids", &SegmentModuleProxy::output_ids);

    // TODO(drobison): need to think about if/how we want to expose type_ids to Python... It might allow for some nice
    // flexibility SegmentModule.def("input_port_type_id", &SegmentModuleProxy::input_port_type_id, py::arg("input_id"))
    // SegmentModule.def("input_port_type_ids", &SegmentModuleProxy::input_port_type_id)
    // SegmentModule.def("output_port_type_id", &SegmentModuleProxy::output_port_type_id, py::arg("output_id"))
    // SegmentModule.def("output_port_type_ids", &SegmentModuleProxy::output_port_type_id)

    /** Module Register Interface Declarations **/
    SegmentModuleRegistry.def(py::init());

    SegmentModuleRegistry.def(
        "contains_namespace", &ModuleRegistryProxy::contains_namespace, py::arg("registry_namespace"));

    SegmentModuleRegistry.def(
        "register_module",
        static_cast<void (*)(ModuleRegistryProxy&,
                             std::string,
                             const std::vector<unsigned int>&,
                             std::function<void(srf::segment::Builder&)>)>(&ModuleRegistryProxy::register_module),
        py::arg("name"),
        py::arg("release_version"),
        py::arg("fn_constructor"));

    SegmentModuleRegistry.def(
        "register_module",
        static_cast<void (*)(ModuleRegistryProxy&,
                             std::string,
                             std::string,
                             const std::vector<unsigned int>&,
                             std::function<void(srf::segment::Builder&)>)>(&ModuleRegistryProxy::register_module),
        py::arg("name"),
        py::arg("registry_namespace"),
        py::arg("release_version"),
        py::arg("fn_constructor"));

    SegmentModuleRegistry.def("unregister_module",
                              &ModuleRegistryProxy::unregister_module,
                              py::arg("name"),
                              py::arg("registry_namespace"),
                              py::arg("optional") = true);

#ifdef VERSION_INFO
    plugins_module.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace srf::pysrf
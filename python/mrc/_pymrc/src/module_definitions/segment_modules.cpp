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

#include "pymrc/module_definitions/segment_modules.hpp"

#include "pymrc/module_registry.hpp"
#include "pymrc/segment_modules.hpp"

#include "mrc/modules/segment_modules.hpp"
#include "mrc/segment/builder.hpp"

#include "nlohmann/json.hpp"

#include <pybind11/pybind11.h>

#include <functional>
#include <map>

namespace mrc::pymrc {

namespace py = pybind11;

class PySegmentModule : public mrc::modules::SegmentModule
{
    using mrc::modules::SegmentModule::SegmentModule;

    void initialize(segment::Builder& builder) override
    {
        PYBIND11_OVERLOAD_PURE(void, mrc::modules::SegmentModule, initialize, builder);
    }

    std::string module_type_name() const override
    {
        PYBIND11_OVERLOAD_PURE(std::string, mrc::modules::SegmentModule, module_type_name);
    }
};

void init_segment_modules(py::module_& module)
{
    auto SegmentModuleRegistry = py::class_<ModuleRegistryProxy>(module, "ModuleRegistry");
    auto SegmentModule =
        py::class_<mrc::modules::SegmentModule, PySegmentModule, std::shared_ptr<mrc::modules::SegmentModule>>(module,
                                                                                                               "Segment"
                                                                                                               "Modul"
                                                                                                               "e");

    /** Segment Module Interface Declarations **/
    SegmentModule.def(py::init<std::string>());

    SegmentModule.def("config", &SegmentModuleProxy::config);

    SegmentModule.def("component_prefix", &SegmentModuleProxy::component_prefix);

    SegmentModule.def("input_port", &SegmentModuleProxy::input_port, py::arg("input_id"));

    SegmentModule.def("input_ports", &SegmentModuleProxy::input_ports);

    SegmentModule.def("module_type_name", &SegmentModuleProxy::module_type_name);

    SegmentModule.def("name", &SegmentModuleProxy::name);

    SegmentModule.def("output_port", &SegmentModuleProxy::output_port, py::arg("output_id"));

    SegmentModule.def("output_ports", &SegmentModuleProxy::output_ports);

    SegmentModule.def("input_ids", &SegmentModuleProxy::input_ids);

    SegmentModule.def("output_ids", &SegmentModuleProxy::output_ids);

    // TODO(devin): need to think about if/how we want to expose type_ids to Python... It might allow for some nice
    // flexibility
    // SegmentModule.def("input_port_type_id", &SegmentModuleProxy::input_port_type_id, py::arg("input_id"))
    // SegmentModule.def("input_port_type_ids", &SegmentModuleProxy::input_port_type_id)
    // SegmentModule.def("output_port_type_id", &SegmentModuleProxy::output_port_type_id, py::arg("output_id"))
    // SegmentModule.def("output_port_type_ids", &SegmentModuleProxy::output_port_type_id)

    /** Module Register Interface Declarations **/
    SegmentModuleRegistry.def_static("contains",
                                     &ModuleRegistryProxy::contains,
                                     py::arg("name"),
                                     py::arg("registry_namespace"));

    SegmentModuleRegistry.def_static("contains_namespace",
                                     &ModuleRegistryProxy::contains_namespace,
                                     py::arg("registry_namespace"));

    SegmentModuleRegistry.def_static("registered_modules", &ModuleRegistryProxy::registered_modules);

    SegmentModuleRegistry.def_static("is_version_compatible",
                                     &ModuleRegistryProxy::is_version_compatible,
                                     py::arg("release_version"));

    SegmentModuleRegistry.def_static("get_module_constructor",
                                     &ModuleRegistryProxy::get_module_constructor,
                                     py::arg("name"),
                                     py::arg("registry_namespace"));

    SegmentModuleRegistry.def_static(
        "register_module",
        static_cast<void (*)(std::string, const std::vector<unsigned int>&, std::function<void(mrc::segment::Builder&)>)>(
            &ModuleRegistryProxy::register_module),
        py::arg("name"),
        py::arg("release_version"),
        py::arg("fn_constructor"));

    SegmentModuleRegistry.def_static(
        "register_module",
        static_cast<void (*)(std::string,
                             std::string,
                             const std::vector<unsigned int>&,
                             std::function<void(mrc::segment::Builder&)>)>(&ModuleRegistryProxy::register_module),
        py::arg("name"),
        py::arg("registry_namespace"),
        py::arg("release_version"),
        py::arg("fn_constructor"));

    SegmentModuleRegistry.def_static("unregister_module",
                                     &ModuleRegistryProxy::unregister_module,
                                     py::arg("name"),
                                     py::arg("registry_namespace"),
                                     py::arg("optional") = true);
}
}  // namespace mrc::pymrc
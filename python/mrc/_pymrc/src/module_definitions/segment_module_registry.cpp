/**
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

#include "pymrc/module_definitions/segment_module_registry.hpp"

#include "pymrc/module_registry.hpp"

#include "mrc/modules/segment_modules.hpp"
#include "mrc/segment/builder.hpp"

#include <nlohmann/json.hpp>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <vector>

namespace mrc::pymrc {

namespace py = pybind11;

void init_segment_module_registry(py::module_& module)
{
    auto SegmentModuleRegistry = py::class_<ModuleRegistryProxy>(module, "ModuleRegistry");

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

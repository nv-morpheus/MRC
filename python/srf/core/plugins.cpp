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

#include "pysrf/plugins.hpp"

#include "pysrf/utils.hpp"

#include "srf/experimental/modules/plugins.hpp"
#include "srf/version.hpp"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <memory>

// IWYU thinks the Segment.def calls need array and vector
// IWYU pragma: no_include <array>
// IWYU pragma: no_include <vector>
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include <pybind11/detail/descr.h>

const std::vector<unsigned int> PybindSegmentModuleVersion{srf_VERSION_MAJOR, srf_VERSION_MINOR, srf_VERSION_PATCH};

namespace srf::pysrf {

namespace py = pybind11;

PYBIND11_MODULE(plugins, module)
{
    module.doc() = R"pbdoc(
        Python bindings for SRF Plugins
        -------------------------------
        .. currentmodule:: plugins
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // Common must be first in every module
    pysrf::import(module, "srf.core.common");

    auto PluginModule = py::class_<srf::modules::PluginModule, std::shared_ptr<srf::modules::PluginModule>>(
        module, "PluginModule");

    /** Module Register Interface Declarations **/
    PluginModule.def("create_or_acquire", &PluginProxy::create_or_acquire, py::return_value_policy::reference_internal);

    PluginModule.def("list_modules", &srf::modules::PluginModule::list_modules);

    PluginModule.def("load", &srf::modules::PluginModule::load, py::arg("throw_on_error") = true);

    PluginModule.def("reload", &srf::modules::PluginModule::reload);

    PluginModule.def("reset_library_directory", &srf::modules::PluginModule::reset_library_directory);

    PluginModule.def("set_library_directory", &srf::modules::PluginModule::set_library_directory, py::arg("path"));

    PluginModule.def("unload", &srf::modules::PluginModule::unload, py::arg("throw_on_error") = true);

    std::stringstream sstream;
    sstream << srf_VERSION_MAJOR << "." << srf_VERSION_MINOR << "." << srf_VERSION_PATCH;

    module.attr("__version__") = sstream.str();

}
}  // namespace srf::pysrf

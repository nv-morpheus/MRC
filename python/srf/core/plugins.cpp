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

#include "srf/modules/plugins.hpp"
#include "srf/utils/string_utils.hpp"
#include "srf/version.hpp"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <memory>
#include <sstream>

// IWYU thinks the Segment.def calls need array and vector
// IWYU pragma: no_include <array>
// IWYU pragma: no_include <vector>
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include <pybind11/detail/descr.h>

const std::vector<unsigned int> PybindSegmentModuleVersion{srf_VERSION_MAJOR, srf_VERSION_MINOR, srf_VERSION_PATCH};

namespace mrc::pysrf {

namespace py = pybind11;

PYBIND11_MODULE(plugins, module)
{
    module.doc() = R"pbdoc(
        Python bindings for MRC Plugins
        -------------------------------
        .. currentmodule:: plugins
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // Common must be first in every module
    pysrf::import(module, "srf.core.common");
    pysrf::import_module_object(module, "srf.core.segment", "SegmentModule");

    auto PluginModule =
        py::class_<mrc::modules::PluginModule, std::shared_ptr<mrc::modules::PluginModule>>(module, "PluginModule");

    /** Module Register Interface Declarations **/
    PluginModule.def("create_or_acquire", &PluginProxy::create_or_acquire, py::return_value_policy::reference_internal);

    PluginModule.def("list_modules", &mrc::modules::PluginModule::list_modules);

    PluginModule.def("load", &mrc::modules::PluginModule::load, py::arg("throw_on_error") = true);

    PluginModule.def("reload", &mrc::modules::PluginModule::reload);

    PluginModule.def("reset_library_directory", &mrc::modules::PluginModule::reset_library_directory);

    PluginModule.def("set_library_directory", &mrc::modules::PluginModule::set_library_directory, py::arg("path"));

    PluginModule.def("unload", &mrc::modules::PluginModule::unload, py::arg("throw_on_error") = true);

    module.attr("__version__") =
        SRF_CONCAT_STR(srf_VERSION_MAJOR << "." << srf_VERSION_MINOR << "." << srf_VERSION_PATCH);
}
}  // namespace mrc::pysrf

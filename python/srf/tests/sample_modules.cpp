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

#include "srf/modules/sample_modules.hpp"

#include "pysrf/utils.hpp"

#include "srf/channel/status.hpp"
#include "srf/modules/module_registry_util.hpp"
#include "srf/node/rx_source.hpp"
#include "srf/utils/string_utils.hpp"
#include "srf/version.hpp"

#include <boost/hana/if.hpp>
#include <pybind11/pybind11.h>

#include <memory>
#include <ostream>

// IWYU thinks the Segment.def calls need array and vector
// IWYU pragma: no_include <array>
// IWYU pragma: no_include <vector>
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include <pybind11/detail/descr.h>

const std::vector<unsigned int> PybindSegmentModuleVersion{srf_VERSION_MAJOR, srf_VERSION_MINOR, srf_VERSION_PATCH};

namespace srf::pysrf {

namespace py = pybind11;

PYBIND11_MODULE(sample_modules, module)
{
    module.doc() = R"pbdoc(
       Python bindings for SRF Unittest Exports
       -------------------------------
       .. currentmodule:: plugins
       .. autosummary::
          :toctree: _generate
   )pbdoc";

    pysrf::import(module, "srf.core.common");

    /** Register test modules -- necessary for python unit tests**/
    modules::ModelRegistryUtil::create_registered_module<srf::modules::SimpleModule>(
        "SimpleModule", "srf_unittest", PybindSegmentModuleVersion);
    modules::ModelRegistryUtil::create_registered_module<srf::modules::ConfigurableModule>(
        "ConfigurableModule", "srf_unittest", PybindSegmentModuleVersion);
    modules::ModelRegistryUtil::create_registered_module<srf::modules::SourceModule>(
        "SourceModule", "srf_unittest", PybindSegmentModuleVersion);
    modules::ModelRegistryUtil::create_registered_module<srf::modules::SinkModule>(
        "SinkModule", "srf_unittest", PybindSegmentModuleVersion);
    modules::ModelRegistryUtil::create_registered_module<srf::modules::NestedModule>(
        "NestedModule", "srf_unittest", PybindSegmentModuleVersion);
    modules::ModelRegistryUtil::create_registered_module<srf::modules::TemplateModule<int>>(
        "TemplateModuleInt", "srf_unittest", PybindSegmentModuleVersion);
    modules::ModelRegistryUtil::create_registered_module<srf::modules::TemplateModule<std::string>>(
        "TemplateModuleString", "srf_unittest", PybindSegmentModuleVersion);

    module.attr("__version__") =
        SRF_CONCAT_STR(srf_VERSION_MAJOR << "." << srf_VERSION_MINOR << "." << srf_VERSION_PATCH);
}
}  // namespace srf::pysrf

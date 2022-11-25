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

#include "mrc/modules/sample_modules.hpp"

#include "pymrc/utils.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/modules/module_registry_util.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <boost/hana/if.hpp>
#include <pybind11/pybind11.h>

#include <memory>
#include <ostream>

// IWYU thinks the Segment.def calls need array and vector
// IWYU pragma: no_include <array>
// IWYU pragma: no_include <vector>
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include <pybind11/detail/descr.h>

const std::vector<unsigned int> PybindSegmentModuleVersion{mrc_VERSION_MAJOR, mrc_VERSION_MINOR, mrc_VERSION_PATCH};

namespace mrc::pymrc {

namespace py = pybind11;

PYBIND11_MODULE(sample_modules, module)
{
    module.doc() = R"pbdoc(
       Python bindings for MRC Unittest Exports
       -------------------------------
       .. currentmodule:: plugins
       .. autosummary::
          :toctree: _generate
   )pbdoc";

    pymrc::import(module, "mrc.core.common");

    /** Register test modules -- necessary for python unit tests**/
    modules::ModelRegistryUtil::create_registered_module<mrc::modules::SimpleModule>(
        "SimpleModule", "mrc_unittest", PybindSegmentModuleVersion);
    modules::ModelRegistryUtil::create_registered_module<mrc::modules::ConfigurableModule>(
        "ConfigurableModule", "mrc_unittest", PybindSegmentModuleVersion);
    modules::ModelRegistryUtil::create_registered_module<mrc::modules::SourceModule>(
        "SourceModule", "mrc_unittest", PybindSegmentModuleVersion);
    modules::ModelRegistryUtil::create_registered_module<mrc::modules::SinkModule>(
        "SinkModule", "mrc_unittest", PybindSegmentModuleVersion);
    modules::ModelRegistryUtil::create_registered_module<mrc::modules::NestedModule>(
        "NestedModule", "mrc_unittest", PybindSegmentModuleVersion);
    modules::ModelRegistryUtil::create_registered_module<mrc::modules::TemplateModule<int>>(
        "TemplateModuleInt", "mrc_unittest", PybindSegmentModuleVersion);
    modules::ModelRegistryUtil::create_registered_module<mrc::modules::TemplateModule<std::string>>(
        "TemplateModuleString", "mrc_unittest", PybindSegmentModuleVersion);

    module.attr("__version__") =
        MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "." << mrc_VERSION_PATCH);
}
}  // namespace mrc::pymrc

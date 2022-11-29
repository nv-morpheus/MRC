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

#include "pymrc/options.hpp"

#include "pymrc/utils.hpp"

#include "mrc/options/engine_groups.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/placement.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/runnable/types.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <pybind11/pybind11.h>

#include <memory>
#include <sstream>

namespace mrc::pymrc {

namespace py = pybind11;

class Config
{};

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(options, module)
{
    module.doc() = R"pbdoc(
        Python bindings for MRC options
        -------------------------------
        .. currentmodule:: options
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // Common must be first in every module
    pymrc::import(module, "mrc.core.common");

    py::class_<Config>(module, "Config")
        .def_property_static("default_channel_size",
                             &ConfigProxy::get_default_channel_size,
                             &ConfigProxy::set_default_channel_size,
                             R"doc(
                Sets the default size of the buffers between edges for all newly created edges. Larger size will reduce backpressure at the cost of memory.
            )doc");

    py::enum_<mrc::PlacementStrategy>(module, "PlacementStrategy")
        .value("PerMachine", mrc::PlacementStrategy::PerMachine)
        .value("PerNumaNode", mrc::PlacementStrategy::PerNumaNode)
        .export_values();

    py::enum_<mrc::runnable::EngineType>(module, "EngineType")
        .value("Fiber", mrc::runnable::EngineType::Fiber)
        .value("Process", mrc::runnable::EngineType::Process)
        .value("Thread", mrc::runnable::EngineType::Thread)
        .export_values();

    py::class_<mrc::TopologyOptions>(module, "TopologyOptions")
        .def(py::init<>())
        .def_property("user_cpuset", &OptionsProxy::get_user_cpuset, &OptionsProxy::set_user_cpuset);

    py::class_<mrc::PlacementOptions>(module, "PlacementOptions")
        .def(py::init<>())
        .def_property("cpu_strategy", &OptionsProxy::get_cpu_strategy, &OptionsProxy::set_cpu_strategy);

    py::class_<mrc::EngineFactoryOptions>(module, "EngineFactoryOptions")
        .def(py::init<>())
        .def_property("cpu_count", &EngineFactoryOptionsProxy::get_cpu_count, &EngineFactoryOptionsProxy::set_cpu_count)
        .def_property(
            "engine_type", &EngineFactoryOptionsProxy::get_engine_type, &EngineFactoryOptionsProxy::set_engine_type)
        .def_property("reusable", &EngineFactoryOptionsProxy::get_reusable, &EngineFactoryOptionsProxy::set_reusable)
        .def_property("allow_overlap",
                      &EngineFactoryOptionsProxy::get_allow_overlap,
                      &EngineFactoryOptionsProxy::set_allow_overlap);

    py::class_<mrc::EngineGroups>(module, "EngineGroups")
        .def(py::init<>())
        .def_property(
            "default_engine_type", &mrc::EngineGroups::default_engine_type, &mrc::EngineGroups::set_default_engine_type)
        .def_property("dedicated_main_thread",
                      &mrc::EngineGroups::dedicated_main_thread,
                      &mrc::EngineGroups::set_dedicated_main_thread)
        .def("set_engine_factory_options",
             py::overload_cast<std::string, EngineFactoryOptions>(&mrc::EngineGroups::set_engine_factory_options))
        .def("engine_group_options",
             &mrc::EngineGroups::engine_group_options,
             py::return_value_policy::reference_internal);

    py::class_<mrc::Options, std::shared_ptr<mrc::Options>>(module, "Options")
        .def(py::init<>())
        .def_property_readonly("placement", &OptionsProxy::get_placement, py::return_value_policy::reference_internal)
        .def_property_readonly("topology", &OptionsProxy::get_topology, py::return_value_policy::reference_internal)
        .def_property_readonly(
            "engine_factories", &OptionsProxy::get_engine_factories, py::return_value_policy::reference_internal)
        .def_property("architect_url",
                      // return a const str
                      static_cast<std::string const& (mrc::Options::*)() const>(&mrc::Options::architect_url),
                      static_cast<void (mrc::Options::*)(std::string)>(&mrc::Options::architect_url));

    module.attr("__version__") =
        MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "." << mrc_VERSION_PATCH);
}
}  // namespace mrc::pymrc

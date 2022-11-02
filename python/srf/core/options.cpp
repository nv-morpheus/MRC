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

#include "pysrf/options.hpp"

#include "pysrf/utils.hpp"

#include "srf/options/engine_groups.hpp"
#include "srf/options/options.hpp"
#include "srf/options/placement.hpp"
#include "srf/options/topology.hpp"
#include "srf/runnable/types.hpp"
#include "srf/version.hpp"

#include <pybind11/pybind11.h>

#include <memory>
#include <sstream>

namespace srf::pysrf {

namespace py = pybind11;

class Config
{};

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(options, module)
{
    module.doc() = R"pbdoc(
        Python bindings for SRF options
        -------------------------------
        .. currentmodule:: options
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // Common must be first in every module
    pysrf::import(module, "srf.core.common");

    py::class_<Config>(module, "Config")
        .def_property_static("default_channel_size",
                             &ConfigProxy::get_default_channel_size,
                             &ConfigProxy::set_default_channel_size,
                             R"doc(
                Sets the default size of the buffers between edges for all newly created edges. Larger size will reduce backpressure at the cost of memory.
            )doc");

    py::enum_<srf::PlacementStrategy>(module, "PlacementStrategy")
        .value("PerMachine", srf::PlacementStrategy::PerMachine)
        .value("PerNumaNode", srf::PlacementStrategy::PerNumaNode)
        .export_values();

    py::enum_<srf::runnable::EngineType>(module, "EngineType")
        .value("Fiber", srf::runnable::EngineType::Fiber)
        .value("Process", srf::runnable::EngineType::Process)
        .value("Thread", srf::runnable::EngineType::Thread)
        .export_values();

    py::class_<srf::TopologyOptions>(module, "TopologyOptions")
        .def(py::init<>())
        .def_property("user_cpuset", &OptionsProxy::get_user_cpuset, &OptionsProxy::set_user_cpuset);

    py::class_<srf::PlacementOptions>(module, "PlacementOptions")
        .def(py::init<>())
        .def_property("cpu_strategy", &OptionsProxy::get_cpu_strategy, &OptionsProxy::set_cpu_strategy);

    py::class_<srf::EngineFactoryOptions>(module, "EngineFactoryOptions")
        .def(py::init<>())
        .def_property("cpu_count", &EngineFactoryOptionsProxy::get_cpu_count, &EngineFactoryOptionsProxy::set_cpu_count)
        .def_property(
            "engine_type", &EngineFactoryOptionsProxy::get_engine_type, &EngineFactoryOptionsProxy::set_engine_type)
        .def_property("reusable", &EngineFactoryOptionsProxy::get_reusable, &EngineFactoryOptionsProxy::set_reusable)
        .def_property("allow_overlap",
                      &EngineFactoryOptionsProxy::get_allow_overlap,
                      &EngineFactoryOptionsProxy::set_allow_overlap);

    py::class_<srf::EngineGroups>(module, "EngineGroups")
        .def(py::init<>())
        .def_property(
            "default_engine_type", &srf::EngineGroups::default_engine_type, &srf::EngineGroups::set_default_engine_type)
        .def_property("dedicated_main_thread",
                      &srf::EngineGroups::dedicated_main_thread,
                      &srf::EngineGroups::set_dedicated_main_thread)
        .def("set_engine_factory_options",
             py::overload_cast<std::string, EngineFactoryOptions>(&srf::EngineGroups::set_engine_factory_options))
        .def("engine_group_options",
             &srf::EngineGroups::engine_group_options,
             py::return_value_policy::reference_internal);

    py::class_<srf::Options, std::shared_ptr<srf::Options>>(module, "Options")
        .def(py::init<>())
        .def_property_readonly("placement", &OptionsProxy::get_placement, py::return_value_policy::reference_internal)
        .def_property_readonly("topology", &OptionsProxy::get_topology, py::return_value_policy::reference_internal)
        .def_property_readonly(
            "engine_factories", &OptionsProxy::get_engine_factories, py::return_value_policy::reference_internal)
        .def_property("architect_url",
                      // return a const str
                      static_cast<std::string const& (srf::Options::*)() const>(&srf::Options::architect_url),
                      static_cast<void (srf::Options::*)(std::string)>(&srf::Options::architect_url));

    std::stringstream sstream;
    sstream << srf_VERSION_MAJOR << "." << srf_VERSION_MINOR << "." << srf_VERSION_PATCH;

    module.attr("__version__") = sstream.str();
}
}  // namespace srf::pysrf

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

#include "pysrf/options.hpp"  // IWYU pragma: associated

#include "srf/channel/channel.hpp"
#include "srf/core/bitmap.hpp"
#include "srf/options/engine_groups.hpp"
#include "srf/options/options.hpp"
#include "srf/options/placement.hpp"
#include "srf/options/topology.hpp"
#include "srf/runnable/types.hpp"

#include <pybind11/pybind11.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>

#include <cstddef>
#include <string>

namespace srf::pysrf {

namespace py = pybind11;

std::size_t ConfigProxy::get_default_channel_size(const py::object& obj)
{
    return srf::channel::default_channel_size();
}

void ConfigProxy::set_default_channel_size(const py::object& obj, size_t default_channel_size)
{
    srf::channel::set_default_channel_size(default_channel_size);
}

std::size_t EngineFactoryOptionsProxy::get_cpu_count(srf::EngineFactoryOptions& self)
{
    return self.cpu_count;
}

void EngineFactoryOptionsProxy::set_cpu_count(srf::EngineFactoryOptions& self, std::size_t cpu_count)
{
    self.cpu_count = cpu_count;
}

runnable::EngineType EngineFactoryOptionsProxy::get_engine_type(srf::EngineFactoryOptions& self)
{
    return self.engine_type;
}

void EngineFactoryOptionsProxy::set_engine_type(srf::EngineFactoryOptions& self, runnable::EngineType engine_type)
{
    self.engine_type = engine_type;
}

bool EngineFactoryOptionsProxy::get_reusable(srf::EngineFactoryOptions& self)
{
    return self.reusable;
}

void EngineFactoryOptionsProxy::set_reusable(srf::EngineFactoryOptions& self, bool reusable)
{
    self.reusable = reusable;
}

bool EngineFactoryOptionsProxy::get_allow_overlap(srf::EngineFactoryOptions& self)
{
    return self.allow_overlap;
}

void EngineFactoryOptionsProxy::set_allow_overlap(srf::EngineFactoryOptions& self, bool allow_overlap)
{
    self.allow_overlap = allow_overlap;
}

std::string OptionsProxy::get_user_cpuset(srf::TopologyOptions& self)
{
    // Convert the CPU set to a string
    return self.user_cpuset().str();
}

void OptionsProxy::set_user_cpuset(srf::TopologyOptions& self, const std::string& user_cpuset)
{
    // Directly set
    self.user_cpuset(user_cpuset);
}

srf::PlacementStrategy OptionsProxy::get_cpu_strategy(srf::PlacementOptions& self)
{
    // Convert the CPU set to a string
    return self.cpu_strategy();
}

void OptionsProxy::set_cpu_strategy(srf::PlacementOptions& self, srf::PlacementStrategy strategy)
{
    // Directly set
    self.cpu_strategy(strategy);
}

srf::PlacementOptions& OptionsProxy::get_placement(srf::Options& self)
{
    return self.placement();
}

srf::TopologyOptions& OptionsProxy::get_topology(srf::Options& self)
{
    return self.topology();
}

srf::EngineGroups& OptionsProxy::get_engine_factories(srf::Options& self)
{
    return self.engine_factories();
}

}  // namespace srf::pysrf

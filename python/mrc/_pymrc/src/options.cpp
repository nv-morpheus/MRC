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

#include "pymrc/options.hpp"  // IWYU pragma: associated

#include "mrc/channel/channel.hpp"
#include "mrc/core/bitmap.hpp"
#include "mrc/options/engine_groups.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/placement.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/runnable/types.hpp"

#include <pybind11/pybind11.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>

#include <cstddef>
#include <string>

namespace mrc::pymrc {

namespace py = pybind11;

std::size_t ConfigProxy::get_default_channel_size(const py::object& obj)
{
    return mrc::channel::default_channel_size();
}

void ConfigProxy::set_default_channel_size(const py::object& obj, size_t default_channel_size)
{
    mrc::channel::set_default_channel_size(default_channel_size);
}

std::size_t EngineFactoryOptionsProxy::get_cpu_count(mrc::EngineFactoryOptions& self)
{
    return self.cpu_count;
}

void EngineFactoryOptionsProxy::set_cpu_count(mrc::EngineFactoryOptions& self, std::size_t cpu_count)
{
    self.cpu_count = cpu_count;
}

runnable::EngineType EngineFactoryOptionsProxy::get_engine_type(mrc::EngineFactoryOptions& self)
{
    return self.engine_type;
}

void EngineFactoryOptionsProxy::set_engine_type(mrc::EngineFactoryOptions& self, runnable::EngineType engine_type)
{
    self.engine_type = engine_type;
}

bool EngineFactoryOptionsProxy::get_reusable(mrc::EngineFactoryOptions& self)
{
    return self.reusable;
}

void EngineFactoryOptionsProxy::set_reusable(mrc::EngineFactoryOptions& self, bool reusable)
{
    self.reusable = reusable;
}

bool EngineFactoryOptionsProxy::get_allow_overlap(mrc::EngineFactoryOptions& self)
{
    return self.allow_overlap;
}

void EngineFactoryOptionsProxy::set_allow_overlap(mrc::EngineFactoryOptions& self, bool allow_overlap)
{
    self.allow_overlap = allow_overlap;
}

std::string OptionsProxy::get_user_cpuset(mrc::TopologyOptions& self)
{
    // Convert the CPU set to a string
    return self.user_cpuset().str();
}

void OptionsProxy::set_user_cpuset(mrc::TopologyOptions& self, const std::string& user_cpuset)
{
    // Directly set
    self.user_cpuset(user_cpuset);
}

mrc::PlacementStrategy OptionsProxy::get_cpu_strategy(mrc::PlacementOptions& self)
{
    // Convert the CPU set to a string
    return self.cpu_strategy();
}

void OptionsProxy::set_cpu_strategy(mrc::PlacementOptions& self, mrc::PlacementStrategy strategy)
{
    // Directly set
    self.cpu_strategy(strategy);
}

mrc::PlacementOptions& OptionsProxy::get_placement(mrc::Options& self)
{
    return self.placement();
}

mrc::TopologyOptions& OptionsProxy::get_topology(mrc::Options& self)
{
    return self.topology();
}

mrc::EngineGroups& OptionsProxy::get_engine_factories(mrc::Options& self)
{
    return self.engine_factories();
}

}  // namespace mrc::pymrc

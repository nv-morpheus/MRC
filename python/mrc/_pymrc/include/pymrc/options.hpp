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

#pragma once

#include "pymrc/forward.hpp"

#include "mrc/options/engine_groups.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/placement.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/runnable/types.hpp"

#include <cstddef>  // for size_t
#include <string>

namespace mrc::pymrc {

// Export everything in the mrc::pymrc namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

class ConfigProxy
{
  public:
    static size_t get_default_channel_size(const pybind11::object& obj);
    static void set_default_channel_size(const pybind11::object& obj, size_t default_channel_size);
};

class EngineFactoryOptionsProxy
{
  public:
    static std::size_t get_cpu_count(mrc::EngineFactoryOptions& self);

    static void set_cpu_count(mrc::EngineFactoryOptions& self, std::size_t cpu_count);

    static runnable::EngineType get_engine_type(mrc::EngineFactoryOptions& self);

    static void set_engine_type(mrc::EngineFactoryOptions& self, runnable::EngineType engine_type);

    static bool get_reusable(mrc::EngineFactoryOptions& self);

    static void set_reusable(mrc::EngineFactoryOptions& self, bool reusable);

    static bool get_allow_overlap(mrc::EngineFactoryOptions& self);

    static void set_allow_overlap(mrc::EngineFactoryOptions& self, bool allow_overlap);
};

class OptionsProxy
{
  public:
    static std::string get_user_cpuset(mrc::TopologyOptions& self);

    static void set_user_cpuset(mrc::TopologyOptions& self, const std::string& user_cpuset);

    static mrc::PlacementStrategy get_cpu_strategy(mrc::PlacementOptions& self);

    static void set_cpu_strategy(mrc::PlacementOptions& self, mrc::PlacementStrategy strategy);

    static mrc::PlacementOptions& get_placement(mrc::Options& self);

    static mrc::TopologyOptions& get_topology(mrc::Options& self);

    static mrc::EngineGroups& get_engine_factories(mrc::Options& self);
};

#pragma GCC visibility pop
}  // namespace mrc::pymrc

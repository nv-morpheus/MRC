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

#include "pysrf/forward.hpp"

#include "srf/options/engine_groups.hpp"
#include "srf/options/options.hpp"
#include "srf/options/placement.hpp"
#include "srf/options/topology.hpp"
#include "srf/runnable/types.hpp"

#include <cstddef>  // for size_t
#include <string>

namespace srf::pysrf {

// Export everything in the srf::pysrf namespace by default since we compile with -fvisibility=hidden
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
    static std::size_t get_cpu_count(srf::EngineFactoryOptions& self);

    static void set_cpu_count(srf::EngineFactoryOptions& self, std::size_t cpu_count);

    static runnable::EngineType get_engine_type(srf::EngineFactoryOptions& self);

    static void set_engine_type(srf::EngineFactoryOptions& self, runnable::EngineType engine_type);

    static bool get_reusable(srf::EngineFactoryOptions& self);

    static void set_reusable(srf::EngineFactoryOptions& self, bool reusable);

    static bool get_allow_overlap(srf::EngineFactoryOptions& self);

    static void set_allow_overlap(srf::EngineFactoryOptions& self, bool allow_overlap);
};

class OptionsProxy
{
  public:
    static std::string get_user_cpuset(srf::TopologyOptions& self);

    static void set_user_cpuset(srf::TopologyOptions& self, const std::string& user_cpuset);

    static srf::PlacementStrategy get_cpu_strategy(srf::PlacementOptions& self);

    static void set_cpu_strategy(srf::PlacementOptions& self, srf::PlacementStrategy strategy);

    static srf::PlacementOptions& get_placement(srf::Options& self);

    static srf::TopologyOptions& get_topology(srf::Options& self);

    static srf::EngineGroups& get_engine_factories(srf::Options& self);
};

#pragma GCC visibility pop
}  // namespace srf::pysrf

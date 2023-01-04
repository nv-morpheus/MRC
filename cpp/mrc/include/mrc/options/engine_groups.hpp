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

#include "mrc/runnable/types.hpp"

#include <cstddef>
#include <functional>
#include <map>
#include <string>

namespace mrc {

// possible global method to get default engine type
// this should be an env or compile time option, not a runtime option

/**
 * @brief Name of the Default EngineFactory
 *
 * @return std::string
 */
extern std::string default_engine_factory_name();

struct EngineFactoryOptions
{
    // number of logical cpus requested
    std::size_t cpu_count{1};

    // engine type
    runnable::EngineType engine_type{runnable::EngineType::Fiber};

    // reusable - if false, logical cpus can only be given to a single set of engines/launcher
    // default is true and the cpus are allocated in a round robin manner
    bool reusable{true};

    // allow overlap - if false, the set of cpus (CpuSet) is guaranteed to be unique across all groups, that is, the
    // intersection with the union of all other groups is the nullset.
    // if true, the CpuSet assigned to this group can have full or partial overlap with other groups
    bool allow_overlap{false};
};

/**
 * @brief Pool of resources selected by name LaunchOptions used to construct one or more Engines for a Runnable
 */
class EngineGroups final
{
  public:
    EngineGroups();

    void set_engine_factory_options(std::string group_name, EngineFactoryOptions options);
    void set_engine_factory_options(std::string group_name, std::function<void(EngineFactoryOptions&)> options_fn);
    void set_dedicated_main_thread(bool default_false);
    void set_dedicated_network_thread(bool default_false);
    void set_default_engine_type(runnable::EngineType engine_type);
    void set_ignore_hyper_threads(bool default_false);

    const EngineFactoryOptions& engine_group_options(const std::string& name) const;
    const std::map<std::string, EngineFactoryOptions>& map() const;
    bool dedicated_main_thread() const;
    bool dedicated_network_thread() const;
    bool ignore_hyper_threads() const;
    runnable::EngineType default_engine_type() const;

  private:
    bool m_dedicated_main_thread{false};
    bool m_dedicated_network_thread{false};
    bool m_ignore_hyper_threads{false};
    runnable::EngineType m_default_engine_type{runnable::EngineType::Fiber};
    std::map<std::string, EngineFactoryOptions> m_engine_resource_groups;
};

}  // namespace mrc

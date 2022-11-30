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
#include "mrc/options/engine_groups.hpp"

#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/runnable/types.hpp"

#include <functional>
#include <map>
#include <string>
#include <utility>

namespace mrc {

std::string default_engine_factory_name()
{
    return std::string("default");
}

EngineGroups::EngineGroups() = default;

void EngineGroups::set_engine_factory_options(std::string group_name, EngineFactoryOptions options)
{
    if (group_name == "main" or group_name == default_engine_factory_name()) {}
    m_engine_resource_groups[group_name] = std::move(options);
}

void EngineGroups::set_engine_factory_options(std::string group_name,
                                              std::function<void(EngineFactoryOptions& options)> options_fn)
{
    EngineFactoryOptions options;
    options_fn(options);
    set_engine_factory_options(std::move(group_name), std::move(options));
}

const EngineFactoryOptions& EngineGroups::engine_group_options(const std::string& name) const
{
    auto search = m_engine_resource_groups.find(name);
    if (search == m_engine_resource_groups.end())
    {
        throw exceptions::MrcRuntimeError("Unknown EngineGroup name: " + name);
    }
    return search->second;
}

void EngineGroups::set_dedicated_main_thread(bool default_false)
{
    m_dedicated_main_thread = default_false;
}

void EngineGroups::set_dedicated_network_thread(bool default_false)
{
    m_dedicated_network_thread = default_false;
}

bool EngineGroups::dedicated_main_thread() const
{
    return m_dedicated_main_thread;
}

bool EngineGroups::dedicated_network_thread() const
{
    return m_dedicated_network_thread;
}

runnable::EngineType EngineGroups::default_engine_type() const
{
    return m_default_engine_type;
}

void EngineGroups::set_default_engine_type(runnable::EngineType engine_type)
{
    m_default_engine_type = engine_type;
}

void EngineGroups::set_ignore_hyper_threads(bool default_false)
{
    m_ignore_hyper_threads = default_false;
}
const std::map<std::string, EngineFactoryOptions>& EngineGroups::map() const
{
    return m_engine_resource_groups;
}
bool EngineGroups::ignore_hyper_threads() const
{
    return m_ignore_hyper_threads;
}
}  // namespace mrc

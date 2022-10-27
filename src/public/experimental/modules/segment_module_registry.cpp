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

#include "srf/experimental/modules/segment_module_registry.hpp"

#include "srf/experimental/modules/segment_modules.hpp"

#include <iostream>
#include <map>
#include <mutex>
#include <vector>

namespace srf::modules {

ModuleRegistry::module_namespace_map_t ModuleRegistry::s_module_namespace_registry{
    {"default", ModuleRegistry::module_registry_map_t{}}};

ModuleRegistry::module_name_map_t ModuleRegistry::s_module_name_map{{"default", std::vector<std::string>{}}};

std::recursive_mutex ModuleRegistry::s_mutex{};

bool ModuleRegistry::contains(const std::string& name, const std::string& registry_namespace)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);

    if (!contains_namespace(registry_namespace))
    {
        return false;
    }

    auto& module_registry = s_module_namespace_registry[registry_namespace];
    auto iter_reg         = module_registry.find(name);

    return iter_reg != module_registry.end();
}

bool ModuleRegistry::contains_namespace(const std::string& registry_namespace)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);

    return s_module_namespace_registry.find(registry_namespace) != s_module_namespace_registry.end();
}

ModuleRegistry::module_constructor_t ModuleRegistry::find_module(const std::string& name,
                                                                 const std::string& registry_namespace)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);

    if (contains(name, registry_namespace))
    {
        return s_module_namespace_registry[registry_namespace][name];
    }

    std::stringstream sstream;

    sstream << "Module does not exist -> " << registry_namespace << "::" << name;
    throw std::invalid_argument(sstream.str());
}

const ModuleRegistry::module_name_map_t& ModuleRegistry::registered_modules()
{
    return ModuleRegistry::s_module_name_map;
}

void ModuleRegistry::register_module(std::string name,
                                     srf::modules::ModuleRegistry::module_constructor_t fn_constructor,
                                     std::string registry_namespace)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);

    if (!contains_namespace(registry_namespace))
    {
        s_module_namespace_registry[registry_namespace] = ModuleRegistry::module_registry_map_t{};
        s_module_name_map[registry_namespace]           = std::vector<std::string>();

        VLOG(2) << "Creating namespace because it does not exist:  " << registry_namespace;
    }

    if (!contains(name, registry_namespace))
    {
        auto& module_registry = s_module_namespace_registry[registry_namespace];
        module_registry[name] = fn_constructor;

        auto& module_name_map = s_module_name_map[registry_namespace];
        module_name_map.push_back(name);

        std::sort(module_name_map.begin(), module_name_map.end());

        VLOG(2) << "Registered module: " << registry_namespace << "::" << name << std::endl;
        return;
    }

    std::stringstream sstream;

    sstream << "Attempt to register duplicate module -> " << registry_namespace << ":" << name;
    throw std::invalid_argument(sstream.str());
}

}  // namespace srf::modules

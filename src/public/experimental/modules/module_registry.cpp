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

#include "srf/experimental/modules/module_registry.hpp"

#include "srf/experimental/modules/segment_modules.hpp"
#include "srf/version.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>

namespace srf::modules {

const unsigned int ModuleRegistry::VersionElements{3};

const std::vector<unsigned int> ModuleRegistry::Version{srf_VERSION_MAJOR, srf_VERSION_MINOR, srf_VERSION_PATCH};

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
    VLOG(2) << sstream.str();
    throw std::invalid_argument(sstream.str());
}

const ModuleRegistry::module_name_map_t& ModuleRegistry::registered_modules()
{
    return ModuleRegistry::s_module_name_map;
}

void ModuleRegistry::register_module(std::string name,
                                     const std::vector<unsigned int>& release_version,
                                     srf::modules::ModuleRegistry::module_constructor_t fn_constructor)
{
    register_module(std::move(name), "default", release_version, fn_constructor);
}

void ModuleRegistry::register_module(std::string name,
                                     std::string registry_namespace,
                                     const std::vector<unsigned int>& release_version,
                                     srf::modules::ModuleRegistry::module_constructor_t fn_constructor)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);
    VLOG(2) << "Registering module: " << registry_namespace << "::" << name;
    if (!is_version_compatible(release_version))
    {
        std::stringstream sstream;
        sstream << "Failed to register module -> module version is: '" << version_to_string(release_version)
                << "' and registry requires: '" << version_to_string(Version);

        throw std::runtime_error(sstream.str());
    }

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
    VLOG(2) << sstream.str();
    throw std::invalid_argument(sstream.str());
}

void ModuleRegistry::unregister_module(const std::string& name, const std::string& registry_namespace, bool optional)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);

    VLOG(2) << "Unregistering module " << registry_namespace << "::" << name;

    if (contains(name, registry_namespace))
    {
        s_module_namespace_registry[registry_namespace].erase(name);

        auto& name_map  = s_module_name_map[registry_namespace];
        auto iter_erase = std::find(name_map.begin(), name_map.end(), name);

        name_map.erase(iter_erase);

        if (s_module_namespace_registry[registry_namespace].empty())
        {
            VLOG(2) << "Namespace " << registry_namespace << " is empty, removing.";
            s_module_namespace_registry.erase(registry_namespace);
            s_module_name_map.erase(registry_namespace);
        }

        return;
    }

    if (optional)
    {
        return;
    }

    std::stringstream sstream;

    sstream << "Failed to unregister module -> " << registry_namespace << "::" << name << " does not exist.";
    VLOG(5) << sstream.str();
    throw std::invalid_argument(sstream.str());
}

bool ModuleRegistry::is_version_compatible(const std::vector<unsigned int>& release_version)
{
    // TODO(devin) improve criteria for module compatibility
    return std::equal(ModuleRegistry::Version.begin(),
                      ModuleRegistry::Version.begin() + ModuleRegistry::VersionElements,
                      release_version.begin());
}

std::string ModuleRegistry::version_to_string(const std::vector<unsigned int>& release_version)
{
    if (release_version.empty())
    {
        return {""};
    }

    std::stringstream sstream;
    sstream << release_version[0];
    std::for_each(release_version.begin() + 1, release_version.end(), [&sstream](unsigned int element) {
        sstream << "." << element;
    });

    return sstream.str();
}

}  // namespace srf::modules

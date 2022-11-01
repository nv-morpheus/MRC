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

#include "srf/segment/object.hpp"

#include <nlohmann/json.hpp>

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace srf::segment {
class Builder;
}

namespace srf::modules {

class SegmentModule;

/**
 * Simple, thread safe, global module registry.
 */
class ModuleRegistry
{
  public:
    using module_constructor_t =
        std::function<std::shared_ptr<srf::modules::SegmentModule>(std::string module_name, nlohmann::json config)>;

    using module_registry_map_t  = std::map<std::string, module_constructor_t>;
    using module_namespace_map_t = std::map<std::string, module_registry_map_t>;
    using module_name_map_t      = std::map<std::string, std::vector<std::string>>;

    ModuleRegistry() = delete;

    /**
     * Return true if the registry contains 'name' else false
     * @param name Name of the module
     * @return boolean indicating if the registry contains the required module.
     */
    static bool contains(const std::string& name, const std::string& registry_namespace = "default");

    /**
     * Return true if the registry contains the namespace, else false
     * @param registry_namespace Namespace name
     * @return boolean indicating if the registry contains the required namespace.
     */
    static bool contains_namespace(const std::string& registry_namespace);

    /**
     * Attempt to retrieve the module constructor for a given module name; throws an error
     * if the given module does not exist.
     * @param name Name of the module
     * @return Module constructor
     */
    static module_constructor_t find_module(const std::string& name, const std::string& registry_namespace = "default");

    /**
     * Retrieve a map of namespace -> registered module name vectors
     * @return Map of namespace -> registered module vector pairs
     */
    static const module_name_map_t& registered_modules();

    /**
     * Simple register call, places the module into the default namespace
     * @param name Name of the module
     * @param fn_constructor Module constructor
     */
    static void register_module(std::string name,
                                const std::vector<unsigned int>& release_version,
                                module_constructor_t fn_constructor);

    /**
     * Attempt to register the provided module constructor for the given name; throws an error
     * if the module already exists.
     * @param name Name of the module
     * @param registry_namespace Namespace where the module `name` should be registered.
     * @param fn_constructor Module constructor
     */
    static void register_module(std::string name,
                                std::string registry_namespace,
                                const std::vector<unsigned int>& release_version,
                                module_constructor_t fn_constructor);

    /**
     * Unregister an existing module
     * @param name Name of the module to un-register
     * @param registry_namespace Namespace where module `name` should reside.
     * @param optional If true, then it is not an error if the module does not exist.
     */
    static void unregister_module(const std::string& name, const std::string& registry_namespace, bool optional = true);

    /**
     * @param release_version vector of unsigned integers corresponding to the version string to check against the
     * registry version.
     * @return true if release version is compatible with registry version, false otherwise.
     */
    static bool is_version_compatible(const std::vector<unsigned int>& release_version);

  private:
    static const unsigned int VersionElements;
    static const std::vector<unsigned int> Version;

    static module_name_map_t s_module_name_map;
    static module_namespace_map_t s_module_namespace_registry;
    static std::recursive_mutex s_mutex;

    static std::string version_to_string(const std::vector<unsigned int>& release_version);
};

}  // namespace srf::modules

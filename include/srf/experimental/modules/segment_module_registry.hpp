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
    using module_constructor_t =
        std::function<std::shared_ptr<srf::modules::SegmentModule>(std::string module_name, nlohmann::json config)>;

  public:
    ModuleRegistry() = delete;

    /**
     * Return true if the registry contains 'name' else false
     * @param name Name of the module
     * @return boolean indicating if the registry contains the required module.
     */
    static bool contains(const std::string& name);

    /**
     * Attempt to retrieve the module constructor for a given module name; throws an error
     * if the given module does not exist.
     * @param name Name of the module
     * @return Module constructor
     */
    static module_constructor_t find_module(const std::string& name);

    /**
     * Attempt to register the provided module constructor for the given name; throws an error
     * if the module already exists.
     * @param name Name of the module
     * @param fn_constructor Module constructor
     */
    static void register_module(std::string name, module_constructor_t fn_constructor);

  private:
    static std::map<std::string, module_constructor_t> s_module_registry;
    static std::recursive_mutex s_mutex;
};

}  // namespace srf::modules

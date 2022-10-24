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

#include "srf/experimental/modules/segment_modules.hpp"
#include "srf/experimental/modules/segment_module_registry.hpp"

#include <map>
#include <mutex>

namespace srf::modules {

void test(){
    auto test = [](){};
}

std::map<std::string, ModuleRegistry::module_constructor_t> ModuleRegistry::s_module_registry{};
std::recursive_mutex ModuleRegistry::s_mutex{};

bool ModuleRegistry::contains(const std::string& name)
{
    auto iter_reg = s_module_registry.find(name);

    return iter_reg != s_module_registry.end();
}

ModuleRegistry::module_constructor_t ModuleRegistry::find_module(const std::string& name)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);

    auto iter_mod = s_module_registry.find(name);
    if (iter_mod != s_module_registry.end())
    {
        return iter_mod->second;
    }

    std::stringstream sstream;

    sstream << "Module does not exist: " << name;
    throw std::invalid_argument(sstream.str());
}

void ModuleRegistry::register_module(std::string name,
                                     srf::modules::ModuleRegistry::module_constructor_t fn_constructor)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);

    if (!ModuleRegistry::contains(name))
    {
        s_module_registry[std::move(name)] = fn_constructor;
        return;
    }

    std::stringstream sstream;

    sstream << "Attempt to register duplicate module: " << name;
    throw std::invalid_argument(sstream.str());
}

}  // namespace srf::modules

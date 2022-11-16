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

#include "srf/modules/segment_modules.hpp"  // IWYU pragma: keep
#include "srf/segment/forward.hpp"

#include <pybind11/pybind11.h>

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace srf::pysrf {

// Export everything in the srf::pysrf namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

class ModuleRegistryProxy
{
    using registry_version_t = std::vector<unsigned int>;

  public:
    ModuleRegistryProxy() = default;

    static bool contains(const std::string& name, const std::string& registry_namespace);

    static bool contains_namespace(const std::string& registry_namespace);

    static std::map<std::string, std::vector<std::string>> registered_modules();

    static bool is_version_compatible(const registry_version_t& release_version);

    static pybind11::cpp_function get_module_constructor(const std::string& name,
                                                         const std::string& registry_namespace);

    static void register_module(std::string name,
                                const registry_version_t& release_version,
                                std::function<void(srf::segment::Builder&)> fn_py_initializer);

    static void register_module(std::string name,
                                std::string registry_namespace,
                                const registry_version_t& release_version,
                                std::function<void(srf::segment::Builder&)> fn_py_initializer);

    static void unregister_module(const std::string& name, const std::string& registry_namespace, bool optional = true);

  private:
    /**
     * When we register python modules, we have to capture a python-land initializer function, which is in turn
     * stored in the ModuleRegistry -- a global static struct. If the registered modules that capture a python
     * function are not unregistered when the python interpreter exits, it will hang, waiting on their ref counts
     * to drop to zero. To ensure this doesn't happen, we register an atexit callback here that forces all python
     * modules to be unregistered when the interpreter is shut down.
     * @param name Name of the module
     * @param registry_namespace Namespace of the module
     */
    static void register_module_cleanup_fn(const std::string& name, const std::string& registry_namespace);
};

#pragma GCC visibility pop
}  // namespace srf::pysrf

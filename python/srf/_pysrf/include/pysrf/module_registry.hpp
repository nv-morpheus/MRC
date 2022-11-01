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
#include "pysrf/py_segment_module.hpp"
#include "pysrf/utils.hpp"

#include "srf/experimental/modules/module_registry.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace srf::pysrf {
namespace py = pybind11;

// Export everything in the srf::pysrf namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

class ModuleRegistryProxy
{
  public:
    using test_t = std::function<std::shared_ptr<srf::modules::SegmentModule>(std::string, py::dict)>;

    ModuleRegistryProxy() = default;

    static bool contains_namespace(ModuleRegistryProxy& self, const std::string& registry_namespace)
    {
        return srf::modules::ModuleRegistry::contains_namespace(registry_namespace);
    }

    static void register_module(ModuleRegistryProxy& self,
                                std::string name,
                                const std::vector<unsigned int>& release_version,
                                PythonSegmentModule::py_initializer_t fn_py_initializer)
    {
        register_module(self, name, "default", release_version, fn_py_initializer);
    }

    static void register_module(ModuleRegistryProxy&,
                                std::string name,
                                std::string registry_namespace,
                                const std::vector<unsigned int>& release_version,
                                PythonSegmentModule::py_initializer_t fn_py_initializer)
    {
        VLOG(2) << "Registering python module: " << registry_namespace << "::" << name;
        auto fn_constructor = [fn_py_initializer](std::string name, nlohmann::json config) {
            auto module             = std::make_shared<PythonSegmentModule>(std::move(name), std::move(config));
            module->m_py_initialize = fn_py_initializer;

            return module;
        };

        srf::modules::ModuleRegistry::register_module(name, registry_namespace, release_version, fn_constructor);

        register_module_cleanup_fn(name, registry_namespace);
    }

    static void unregister_module(ModuleRegistryProxy& self,
                                  const std::string& name,
                                  const std::string& registry_namespace,
                                  bool optional = true)
    {
        return srf::modules::ModuleRegistry::unregister_module(name, registry_namespace, optional);
    }

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
    static void register_module_cleanup_fn(const std::string& name, const std::string& registry_namespace)
    {
        auto at_exit = pybind11::module_::import("atexit");
        at_exit.attr("register")(pybind11::cpp_function([name, registry_namespace]() {
            VLOG(2) << "(atexit) Unregistering " << registry_namespace << "::" << name;

            // Try unregister -- ignore if already unregistered
            srf::modules::ModuleRegistry::unregister_module(name, registry_namespace, true);
        }));
    }
};

#pragma GCC visibility pop
}  // namespace srf::pysrf
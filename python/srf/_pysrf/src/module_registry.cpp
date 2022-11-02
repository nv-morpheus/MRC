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

#include "pysrf/module_registry.hpp"

#include "pysrf/py_segment_module.hpp"

#include "srf/experimental/modules/module_registry.hpp"

#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <array>
#include <memory>
#include <ostream>
#include <utility>

namespace srf::pysrf {

bool ModuleRegistryProxy::contains(ModuleRegistryProxy& self,
                                   const std::string& name,
                                   const std::string& registry_namespace)
{
    return srf::modules::ModuleRegistry::contains(name, registry_namespace);
}

bool ModuleRegistryProxy::contains_namespace(srf::pysrf::ModuleRegistryProxy& self,
                                             const std::string& registry_namespace)
{
    return srf::modules::ModuleRegistry::contains_namespace(registry_namespace);
}

void ModuleRegistryProxy::register_module(srf::pysrf::ModuleRegistryProxy& self,
                                          std::string name,
                                          const std::vector<unsigned int>& release_version,
                                          std::function<void(srf::segment::Builder&)> fn_py_initializer)
{
    register_module(self, name, "default", release_version, fn_py_initializer);
}

void ModuleRegistryProxy::register_module(srf::pysrf::ModuleRegistryProxy& self,
                                          std::string name,
                                          std::string registry_namespace,
                                          const std::vector<unsigned int>& release_version,
                                          std::function<void(srf::segment::Builder&)> fn_py_initializer)
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

void ModuleRegistryProxy::unregister_module(srf::pysrf::ModuleRegistryProxy& self,
                                            const std::string& name,
                                            const std::string& registry_namespace,
                                            bool optional)
{
    return srf::modules::ModuleRegistry::unregister_module(name, registry_namespace, optional);
}

void ModuleRegistryProxy::register_module_cleanup_fn(const std::string& name, const std::string& registry_namespace)
{
    auto at_exit = pybind11::module_::import("atexit");
    at_exit.attr("register")(pybind11::cpp_function([name, registry_namespace]() {
        VLOG(2) << "(atexit) Unregistering " << registry_namespace << "::" << name;

        // Try unregister -- ignore if already unregistered
        srf::modules::ModuleRegistry::unregister_module(name, registry_namespace, true);
    }));
}

}  // namespace srf::pysrf
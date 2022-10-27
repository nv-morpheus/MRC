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

#include "srf/version.hpp"
#include "srf/experimental/modules/sample_modules.hpp"

#include "test_segment.hpp"

TEST_F(SegmentTests, RegistryModuleTest)
{
    using namespace modules;

    const auto *registry_namespace = "module_registry_unittest";
    const auto *simple_mod_name = "SimpleModule";
    const auto *configurable_mod_name = "ConfigurableModule";

    auto config = nlohmann::json();
    config["source_count"] = 42;

    const std::vector<unsigned int> release_version = {srf_VERSION_MAJOR, srf_VERSION_MINOR, srf_VERSION_PATCH};

    auto simple_mod_func = [](std::string module_name, nlohmann::json config) {
                return std::make_shared<SimpleModule>(std::move(module_name), std::move(config));
            };

    auto configurable_mod_func = [](std::string module_name, nlohmann::json config) {
                return std::make_shared<ConfigurableModule>(std::move(module_name), std::move(config));
            };
    ModuleRegistry::register_module(simple_mod_name, release_version, simple_mod_func);
    ModuleRegistry::register_module(configurable_mod_name, release_version, configurable_mod_func);
    ModuleRegistry::register_module(simple_mod_name, registry_namespace, release_version, simple_mod_func);
    ModuleRegistry::register_module(configurable_mod_name, registry_namespace, release_version, configurable_mod_func);

    // Registering duplicate module throws an exception.
    EXPECT_THROW(ModuleRegistry::register_module(simple_mod_name, release_version, simple_mod_func), std::invalid_argument);
    EXPECT_THROW(ModuleRegistry::register_module(configurable_mod_name, release_version, simple_mod_func), std::invalid_argument);
    EXPECT_THROW(ModuleRegistry::register_module(simple_mod_name, registry_namespace, release_version, simple_mod_func),
                                                 std::invalid_argument);
    EXPECT_THROW(ModuleRegistry::register_module(configurable_mod_name, registry_namespace, release_version, simple_mod_func),
                                                 std::invalid_argument);
}

TEST_F(SegmentTests, ContainsNamespaceTest)
{
    using namespace modules;

    const auto *registry_namespace = "module_registry_unittest";
    const auto *registry_namespace_2 = "module_registry_unittest2";
    const auto *module_name = "SimpleModule";

    auto hasNamespace = ModuleRegistry::contains_namespace(registry_namespace);
    auto hasNamespace_2 = ModuleRegistry::contains_namespace(registry_namespace_2);

    EXPECT_EQ(hasNamespace, true);
    EXPECT_EQ(hasNamespace_2, false);

}

TEST_F(SegmentTests, ContainsModuleTest)
{
    using namespace modules;

    const auto *registry_namespace = "module_registry_unittest";
    const auto *registry_namespace_2 = "module_registry_unittest2";
    const auto *module_name = "SimpleModule";

    auto hasModule = ModuleRegistry::contains(module_name, registry_namespace);
    auto hasModule_2 = ModuleRegistry::contains(module_name, registry_namespace_2);

    EXPECT_EQ(hasModule, true);
    EXPECT_EQ(hasModule_2, false);

}

TEST_F(SegmentTests, FindModuleTest)
{
    using namespace modules;

    const auto *registry_namespace = "module_registry_unittest";
    const auto *module_name = "SimpleModule";
    const auto *module_name_3 = "SimpleModuleTest";

    auto fn_module_constructor = ModuleRegistry::find_module(module_name, registry_namespace);

    // Finding a module that does not exists in the registry throws an exception.
    EXPECT_THROW(ModuleRegistry::find_module(module_name_3), std::invalid_argument);
    EXPECT_THROW(ModuleRegistry::find_module(module_name_3, registry_namespace), std::invalid_argument);
}

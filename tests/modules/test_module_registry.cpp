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

#include "test_modules.hpp"

#include "srf/modules/module_registry.hpp"
#include "srf/modules/plugins.hpp"
#include "srf/modules/sample_modules.hpp"
#include "srf/version.hpp"

#include <boost/filesystem.hpp>
#include <dlfcn.h>

TEST_F(TestModuleRegistry, RegistryModuleTest)
{
    using namespace modules;

    const auto* registry_namespace    = "module_registry_unittest";
    const auto* simple_mod_name       = "SimpleModule";
    const auto* configurable_mod_name = "ConfigurableModule";

    auto config            = nlohmann::json();
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
    EXPECT_THROW(ModuleRegistry::register_module(simple_mod_name, release_version, simple_mod_func),
                 std::invalid_argument);
    EXPECT_THROW(ModuleRegistry::register_module(configurable_mod_name, release_version, simple_mod_func),
                 std::invalid_argument);
    EXPECT_THROW(ModuleRegistry::register_module(simple_mod_name, registry_namespace, release_version, simple_mod_func),
                 std::invalid_argument);
    EXPECT_THROW(
        ModuleRegistry::register_module(configurable_mod_name, registry_namespace, release_version, simple_mod_func),
        std::invalid_argument);
}

TEST_F(TestModuleRegistry, ContainsNamespaceTest)
{
    using namespace modules;

    const auto* registry_namespace   = "module_registry_unittest";
    const auto* registry_namespace_2 = "module_registry_unittest2";
    const auto* module_name          = "SimpleModule";

    auto hasNamespace   = ModuleRegistry::contains_namespace(registry_namespace);
    auto hasNamespace_2 = ModuleRegistry::contains_namespace(registry_namespace_2);

    EXPECT_EQ(hasNamespace, true);
    EXPECT_EQ(hasNamespace_2, false);
}

TEST_F(TestModuleRegistry, ContainsModuleTest)
{
    using namespace modules;

    const auto* registry_namespace   = "module_registry_unittest";
    const auto* registry_namespace_2 = "module_registry_unittest2";
    const auto* module_name          = "SimpleModule";

    auto hasModule   = ModuleRegistry::contains(module_name, registry_namespace);
    auto hasModule_2 = ModuleRegistry::contains(module_name, registry_namespace_2);

    EXPECT_EQ(hasModule, true);
    EXPECT_EQ(hasModule_2, false);
}

TEST_F(TestModuleRegistry, FindModuleTest)
{
    using namespace modules;

    const auto* registry_namespace = "module_registry_unittest";
    const auto* module_name        = "SimpleModule";
    const auto* module_name_3      = "SimpleModuleTest";

    auto fn_module_constructor = ModuleRegistry::get_module_constructor(module_name, registry_namespace);

    // Finding a module that does not exists in the registry throws an exception.
    EXPECT_THROW(ModuleRegistry::get_module_constructor(module_name_3), std::invalid_argument);
    EXPECT_THROW(ModuleRegistry::get_module_constructor(module_name_3, registry_namespace), std::invalid_argument);
}

TEST_F(TestModuleRegistry, UnRegistrerModuleTest)
{
    using namespace modules;

    std::string period              = ".";
    std::string release_version_str = std::to_string(srf_VERSION_MAJOR) + period + std::to_string(srf_VERSION_MINOR) +
                                      period + std::to_string(srf_VERSION_PATCH);

    std::string registry_namespace = "module_registry_unittest";
    std::string simple_mod_name    = "SimpleModule";

    ModuleRegistry::unregister_module(simple_mod_name, release_version_str);

    ModuleRegistry::unregister_module(simple_mod_name, release_version_str, true);

    // Throws an exception when there is no registered module.
    EXPECT_THROW(ModuleRegistry::unregister_module(simple_mod_name, release_version_str, false), std::invalid_argument);
}

TEST_F(TestModuleRegistry, VersionCompatibleTest)
{
    using namespace modules;

    const std::vector<unsigned int> release_version     = {srf_VERSION_MAJOR, srf_VERSION_MINOR, srf_VERSION_PATCH};
    const std::vector<unsigned int> old_release_version = {22, 10, 0};
    const std::vector<unsigned int> no_version_patch    = {22, 10};
    const std::vector<unsigned int> no_version_minor_and_patch = {22};

    EXPECT_EQ(ModuleRegistry::is_version_compatible(release_version), true);
    EXPECT_EQ(ModuleRegistry::is_version_compatible(old_release_version), false);
    EXPECT_EQ(ModuleRegistry::is_version_compatible(no_version_patch), false);
    EXPECT_EQ(ModuleRegistry::is_version_compatible(no_version_minor_and_patch), false);
}

TEST_F(TestModuleRegistry, RegisteredModulesTest)
{
    using namespace modules;

    auto rigestered_mods_map = ModuleRegistry::registered_modules();

    EXPECT_EQ(rigestered_mods_map.size(), 2);
    EXPECT_EQ(rigestered_mods_map.find("default") != rigestered_mods_map.end(), true);
    EXPECT_EQ(rigestered_mods_map.find("module_registry_unittest") != rigestered_mods_map.end(), true);
}

std::string get_modules_path()
{
    int pid = getpid();
    std::stringstream sstream;
    sstream << "/proc/" << pid << "/exe";

    std::string link_id         = sstream.str();
    unsigned int sz_path_buffer = 8102;
    std::vector<char> path_buffer(sz_path_buffer + 1);
    readlink(link_id.c_str(), path_buffer.data(), sz_path_buffer);

    boost::filesystem::path whereami(path_buffer.data());

    std::string modules_path = whereami.parent_path().string() + "/modules/";

    return modules_path;
}

TEST_F(TestModuleRegistry, DynamicModuleLoadTest)
{
    void* module_handle;
    bool (*dummy_entrypoint)();

    std::string module_path = get_modules_path() + "libdynamic_test_module.so";

    module_handle = dlopen(module_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (module_handle == nullptr)
    {
        std::cerr << "Error: " << dlerror() << std::endl;
    }
    EXPECT_TRUE(module_handle);

    dummy_entrypoint        = (bool (*)())dlsym(module_handle, "SRF_MODULE_dummy_entrypoint");
    const char* dlsym_error = dlerror();
    if (dlsym_error != nullptr)
    {
        std::cerr << "Error: " << dlsym_error << std::endl;
    }
    EXPECT_TRUE(dlsym_error == nullptr);
    EXPECT_TRUE(dummy_entrypoint());
}

TEST_F(TestModuleRegistry, DynamicModuleRegistrationTest)
{
    using namespace srf::modules;
    void* module_handle;
    bool (*entrypoint_load)();
    bool (*entrypoint_unload)();

    std::string module_path = get_modules_path() + "libdynamic_test_module.so";

    module_handle = dlopen(module_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (module_handle == nullptr)
    {
        std::cerr << "Error: " << dlerror() << std::endl;
    }
    EXPECT_TRUE(module_handle);

    entrypoint_load         = (bool (*)())dlsym(module_handle, "SRF_MODULE_entrypoint_load");
    const char* dlsym_error = dlerror();
    if (dlsym_error != nullptr)
    {
        std::cerr << "Error: " << dlsym_error << std::endl;
    }
    EXPECT_TRUE(dlsym_error == nullptr);
    EXPECT_TRUE(entrypoint_load());

    std::string module_namespace{"srf_unittest_cpp_dynamic"};
    std::string module_name{"DynamicSourceModule"};

    EXPECT_TRUE(ModuleRegistry::contains_namespace(module_namespace));
    EXPECT_TRUE(ModuleRegistry::contains(module_name, module_namespace));

    entrypoint_unload              = (bool (*)())dlsym(module_handle, "SRF_MODULE_entrypoint_unload");
    const char* dlsym_unload_error = dlerror();
    if (dlsym_unload_error != nullptr)
    {
        std::cerr << "Error: " << dlsym_unload_error << std::endl;
    }
    EXPECT_TRUE(dlsym_unload_error == nullptr);

    unsigned int packet_count{0};

    auto init_wrapper = [&packet_count](segment::Builder& builder) {
        auto config = nlohmann::json();
        unsigned int source_count{42};
        config["source_count"] = source_count;

        auto dynamic_source_mod = builder.load_module_from_registry(
            "DynamicSourceModule", "srf_unittest_cpp_dynamic", "DynamicModuleSourceTest_mod1", config);

        auto sink = builder.make_sink<bool>("sink", [&packet_count](bool input) {
            packet_count++;
            VLOG(10) << "Sinking " << input << std::endl;
        });

        builder.make_edge(dynamic_source_mod->output_port("source"), sink);
    };

    m_pipeline->make_segment("DynamicSourceModule_Segment", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(packet_count, 42);
    EXPECT_TRUE(entrypoint_unload());
    dlclose(module_handle);
}

TEST_F(TestModuleRegistry, DynamicModulePluginInterfaceTest)
{
    using namespace srf::modules;

    auto plugin = PluginModule::create_or_acquire("libdynamic_test_module.so");
    plugin->set_library_directory(get_modules_path());
    plugin->load();
    plugin->reload();

    auto plugin_copy = PluginModule::create_or_acquire("libdynamic_test_module.so");
    // Should be 1 in the global plugin tracker, 1 for plugin, and 1 for plugin_copy
    EXPECT_TRUE(plugin_copy.use_count() == 3);
    plugin_copy.reset();

    EXPECT_TRUE(plugin.use_count() == 2);
}

TEST_F(TestModuleRegistry, DynamicModulePluginRegistrationTest)
{
    using namespace srf::modules;

    // auto plugin = std::unique_ptr<PluginModule>{};
    auto plugin = PluginModule::create_or_acquire("libdynamic_test_module.so");
    plugin->set_library_directory(get_modules_path());
    plugin->load();
    plugin->reload();

    std::string module_namespace{"srf_unittest_cpp_dynamic"};
    std::string module_name{"DynamicSourceModule"};

    EXPECT_TRUE(ModuleRegistry::contains_namespace(module_namespace));
    EXPECT_TRUE(ModuleRegistry::contains(module_name, module_namespace));

    /*
     * The dynamic_test_module registers DynamicSourceModule in three test namespaces:
     * srf_unittest_cpp_dynamic[1|2|3]. Double check this here.
     */
    auto registered_modules = ModuleRegistry::registered_modules();

    EXPECT_TRUE(registered_modules.find("srf_unittest_cpp_dynamic") != registered_modules.end());
    auto& ns_1 = registered_modules["srf_unittest_cpp_dynamic"];
    EXPECT_EQ(ns_1.size(), 1);
    EXPECT_TRUE(ns_1[0] == "DynamicSourceModule");

    EXPECT_TRUE(registered_modules.find("srf_unittest_cpp_dynamic_2") != registered_modules.end());
    auto& ns_2 = registered_modules["srf_unittest_cpp_dynamic_2"];
    EXPECT_EQ(ns_2.size(), 1);
    EXPECT_TRUE(ns_2[0] == "DynamicSourceModule");

    EXPECT_TRUE(registered_modules.find("srf_unittest_cpp_dynamic_3") != registered_modules.end());
    auto& ns_3 = registered_modules["srf_unittest_cpp_dynamic_3"];
    EXPECT_EQ(ns_3.size(), 1);
    EXPECT_TRUE(ns_3[0] == "DynamicSourceModule");

    std::vector<std::string> expected_modules{
        "srf_unittest_cpp_dynamic::DynamicSourceModule",
        "srf_unittest_cpp_dynamic_2::DynamicSourceModule",
        "srf_unittest_cpp_dynamic_3::DynamicSourceModule",
    };

    auto actual_modules = plugin->list_modules();
    EXPECT_EQ(actual_modules.size(), 3);
    EXPECT_TRUE(std::equal(expected_modules.begin(), expected_modules.begin() + 3, actual_modules.begin()));

    plugin->unload();
    registered_modules = ModuleRegistry::registered_modules();

    EXPECT_TRUE(registered_modules.find("srf_unittest_cpp_dynamic") == registered_modules.end());
    EXPECT_TRUE(registered_modules.find("srf_unittest_cpp_dynamic_2") == registered_modules.end());
    EXPECT_TRUE(registered_modules.find("srf_unittest_cpp_dynamic_3") == registered_modules.end());
}

TEST_F(TestModuleRegistry, DynamicModuleBadVersionTest)
{
    using namespace srf::modules;
    void* module_handle;
    bool (*entrypoint)();

    std::string module_path = get_modules_path() + "libdynamic_test_module.so";

    module_handle = dlopen(module_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (module_handle == nullptr)
    {
        std::cerr << "Error: " << dlerror() << std::endl;
    }
    EXPECT_TRUE(module_handle);

    entrypoint              = (bool (*)())dlsym(module_handle, "SRF_MODULE_bad_version_entrypoint");
    const char* dlsym_error = dlerror();
    if (dlsym_error != nullptr)
    {
        std::cerr << "Error: " << dlsym_error << std::endl;
    }
    EXPECT_TRUE(dlsym_error == nullptr);
    EXPECT_THROW(entrypoint(), std::runtime_error);

    std::string module_namespace{"srf_unittest_cpp_dynamic_BAD"};
    std::string module_name{"DynamicSourceModule_BAD"};

    EXPECT_FALSE(ModuleRegistry::contains_namespace(module_namespace));
    EXPECT_FALSE(ModuleRegistry::contains(module_name, module_namespace));
}

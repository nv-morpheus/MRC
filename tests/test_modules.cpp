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

#include "test_segment.hpp"

#include "srf/core/executor.hpp"
#include "srf/engine/pipeline/ipipeline.hpp"
#include "srf/experimental/modules/module_registry.hpp"
#include "srf/experimental/modules/plugins.hpp"
#include "srf/experimental/modules/sample_modules.hpp"
#include "srf/options/options.hpp"
#include "srf/segment/builder.hpp"

#include <boost/filesystem.hpp>
#include <dlfcn.h>
#include <gtest/gtest-message.h>
#include <gtest/gtest-test-part.h>
#include <rxcpp/rx-subscriber.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

TEST_F(SegmentTests, ModuleConstructorTest)
{
    using namespace modules;

    auto config_1            = nlohmann::json();
    auto config_2            = nlohmann::json();
    config_2["config_key_1"] = true;

    auto mod1 = SimpleModule("InitModuleTest_mod1");
    auto mod2 = ConfigurableModule("InitModuleTest_2");
    auto mod3 = ConfigurableModule("InitModuleTest_3", config_1);
    auto mod4 = ConfigurableModule("InitModuleTest_4", config_2);

    ASSERT_EQ(mod4.config().contains("config_key_1"), true);
}

TEST_F(SegmentTests, ModuleInitializationTest)
{
    using namespace modules;

    auto init_wrapper = [](segment::Builder& builder) {
        auto config_1            = nlohmann::json();
        auto config_2            = nlohmann::json();
        config_2["config_key_1"] = true;

        auto simple_mod         = builder.make_module<SimpleModule>("ModuleInitializationTest_mod1");
        auto configurable_1_mod = builder.make_module<ConfigurableModule>("ModuleInitializationTest_mod2", config_1);
        auto configurable_2_mod = builder.make_module<ConfigurableModule>("ModuleInitializationTest_mod3", config_2);
        auto configurable_mod_3 = ConfigurableModule("ModuleInitializationTest_mod4", config_2);

        configurable_mod_3(builder);

        EXPECT_EQ(simple_mod->input_ids().size(), 2);
        EXPECT_EQ(simple_mod->output_ids().size(), 2);
        EXPECT_EQ(simple_mod->input_ports().size(), 2);
        EXPECT_EQ(simple_mod->output_ports().size(), 2);
        EXPECT_EQ(simple_mod->input_ports().find("input1") != simple_mod->input_ports().end(), true);
        EXPECT_EQ(simple_mod->input_ports().find("input2") != simple_mod->input_ports().end(), true);
        EXPECT_EQ(simple_mod->input_port_type_id("input1"), typeid(bool));
        EXPECT_EQ(simple_mod->input_port_type_id("input2"), typeid(bool));
        EXPECT_EQ(simple_mod->input_port_type_ids().find("input1")->second, typeid(bool));
        EXPECT_EQ(simple_mod->input_port_type_ids().find("input2")->second, typeid(bool));
        EXPECT_EQ(simple_mod->output_ports().find("output1") != simple_mod->input_ports().end(), true);
        EXPECT_EQ(simple_mod->output_ports().find("output2") != simple_mod->input_ports().end(), true);
        EXPECT_EQ(simple_mod->output_port_type_id("output1"), typeid(std::string));
        EXPECT_EQ(simple_mod->output_port_type_id("output2"), typeid(std::string));
        EXPECT_EQ(simple_mod->output_port_type_ids().find("output1")->second, typeid(std::string));
        EXPECT_EQ(simple_mod->output_port_type_ids().find("output2")->second, typeid(std::string));

        EXPECT_THROW(simple_mod->input_port("DOES_NOT_EXIST"), std::invalid_argument);
        EXPECT_THROW(simple_mod->output_port("DOES_NOT_EXIST"), std::invalid_argument);
        EXPECT_THROW(simple_mod->input_port_type_id("DOES_NOT_EXIST"), std::invalid_argument);
        EXPECT_THROW(simple_mod->output_port_type_id("DOES_NOT_EXIST"), std::invalid_argument);

        EXPECT_EQ(configurable_1_mod->input_ports().size(), 1);
        EXPECT_EQ(configurable_1_mod->output_ports().size(), 1);
        EXPECT_EQ(configurable_1_mod->m_was_configured, false);

        EXPECT_EQ(configurable_2_mod->input_ports().size(), 1);
        EXPECT_EQ(configurable_2_mod->output_ports().size(), 1);
        EXPECT_EQ(configurable_2_mod->m_was_configured, true);
    };

    m_pipeline->make_segment("Initialization_Segment", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.stop();
    executor.join();
}

TEST_F(SegmentTests, ModuleEndToEndTest)
{
    using namespace modules;
    unsigned int packets_1{0};
    unsigned int packets_2{0};
    unsigned int packets_3{0};

    auto init_wrapper = [&packets_1, &packets_2, &packets_3](segment::Builder& builder) {
        auto simple_mod       = builder.make_module<SimpleModule>("ModuleEndToEndTest_mod1");
        auto configurable_mod = builder.make_module<ConfigurableModule>("ModuleEndToEndTest_mod2");

        auto source1 = builder.make_source<bool>("src1", [](rxcpp::subscriber<bool>& sub) {
            if (sub.is_subscribed())
            {
                sub.on_next(true);
                sub.on_next(false);
                sub.on_next(true);
                sub.on_next(true);
            }

            sub.on_completed();
        });

        // Ex1. Partially dynamic edge construction
        builder.make_edge(source1, simple_mod->input_port("input1"));

        auto source2 = builder.make_source<bool>("src2", [](rxcpp::subscriber<bool>& sub) {
            if (sub.is_subscribed())
            {
                sub.on_next(true);
                sub.on_next(false);
                sub.on_next(false);
                sub.on_next(false);
                sub.on_next(true);
                sub.on_next(false);
            }

            sub.on_completed();
        });

        // Ex2. Dynamic edge construction -- requires type specification
        builder.make_dynamic_edge<bool, bool>(source2, simple_mod->input_port("input2"));

        auto sink1 = builder.make_sink<std::string>("sink1", [&packets_1](std::string input) {
            packets_1++;
            VLOG(10) << "Sinking " << input << std::endl;
        });

        builder.make_edge(simple_mod->output_port("output1"), sink1);

        auto sink2 = builder.make_sink<std::string>("sink2", [&packets_2](std::string input) {
            packets_2++;
            VLOG(10) << "Sinking " << input << std::endl;
        });

        builder.make_edge(simple_mod->output_port("output2"), sink2);

        auto source3 = builder.make_source<bool>("src3", [](rxcpp::subscriber<bool>& sub) {
            if (sub.is_subscribed())
            {
                sub.on_next(true);
                sub.on_next(false);
                sub.on_next(true);
                sub.on_next(true);
            }

            sub.on_completed();
        });

        builder.make_edge(source3, configurable_mod->input_port("configurable_input_a"));

        auto sink3 = builder.make_sink<std::string>("sink3", [&packets_3](std::string input) {
            packets_3++;
            VLOG(10) << "Sinking " << input << std::endl;
        });

        builder.make_edge(configurable_mod->output_port("configurable_output_x"), sink3);
    };

    m_pipeline->make_segment("EndToEnd_Segment", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(packets_1, 4);
    EXPECT_EQ(packets_2, 6);
    EXPECT_EQ(packets_3, 4);
}

TEST_F(SegmentTests, ModuleAsSourceTest)
{
    using namespace modules;

    unsigned int packet_count{0};

    auto init_wrapper = [&packet_count](segment::Builder& builder) {
        auto config = nlohmann::json();
        unsigned int source_count{42};
        config["source_count"] = source_count;

        auto source_mod = builder.make_module<SourceModule>("ModuleSourceTest_mod1", config);

        auto sink = builder.make_sink<bool>("sink", [&packet_count](bool input) {
            packet_count++;
            VLOG(10) << "Sinking " << input << std::endl;
        });

        builder.make_edge(source_mod->output_port("source"), sink);
    };

    m_pipeline->make_segment("SimpleModule_Segment", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(packet_count, 42);
}

TEST_F(SegmentTests, ModuleAsSinkTest)
{
    using namespace modules;

    unsigned int packet_count{0};

    auto init_wrapper = [&packet_count](segment::Builder& builder) {
        auto source = builder.make_source<bool>("source", [&packet_count](rxcpp::subscriber<bool>& sub) {
            if (sub.is_subscribed())
            {
                for (unsigned int i = 0; i < 43; ++i)
                {
                    sub.on_next(true);
                    packet_count++;
                }
            }

            sub.on_completed();
        });

        auto sink_mod = builder.make_module<SinkModule>("ModuleSinkTest_mod1");

        builder.make_edge(source, sink_mod->input_port("sink"));
    };

    m_pipeline->make_segment("SimpleModule_Segment", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(packet_count, 43);
}

TEST_F(SegmentTests, ModuleChainingTest)
{
    using namespace modules;

    auto sink_mod     = std::make_shared<SinkModule>("ModuleChainingTest_mod2");
    auto init_wrapper = [&sink_mod](segment::Builder& builder) {
        auto config = nlohmann::json();
        unsigned int source_count{42};
        config["source_count"] = source_count;

        auto source_mod = builder.make_module<SourceModule>("ModuleChainingTest_mod1", config);
        builder.init_module(sink_mod);

        builder.make_dynamic_edge<bool, bool>(source_mod->output_port("source"), sink_mod->input_port("sink"));
    };

    m_pipeline->make_segment("SimpleModule_Segment", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(sink_mod->m_packet_count, 42);
}

TEST_F(SegmentTests, ModuleNestingTest)
{
    using namespace modules;

    unsigned int packet_count{0};

    auto init_wrapper = [&packet_count](segment::Builder& builder) {
        auto nested_mod = builder.make_module<NestedModule>("ModuleNestingTest_mod1");

        auto nested_sink = builder.make_sink<std::string>("nested_sink", [&packet_count](std::string input) {
            packet_count++;
            VLOG(10) << "Sinking " << input << std::endl;
        });

        builder.make_edge(nested_mod->output_port("nested_module_output"), nested_sink);
    };

    m_pipeline->make_segment("SimpleModule_Segment", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(packet_count, 4);
}

TEST_F(SegmentTests, ModuleTemplateTest)
{
    using namespace modules;

    unsigned int packet_count_1{0};
    unsigned int packet_count_2{0};

    auto init_wrapper = [&packet_count_1, &packet_count_2](segment::Builder& builder) {
        using data_type_1_t = int;
        using data_type_2_t = std::string;

        auto config_1 = nlohmann::json();
        auto config_2 = nlohmann::json();

        unsigned int source_count_1{42};
        unsigned int source_count_2{24};

        config_1["source_count"] = source_count_1;
        config_2["source_count"] = source_count_2;

        auto source_1_mod = builder.make_module<TemplateModule<data_type_1_t>>("ModuleTemplateTest_mod1", config_1);

        auto sink_1 = builder.make_sink<data_type_1_t>("sink_1", [&packet_count_1](data_type_1_t input) {
            packet_count_1++;
            VLOG(10) << "Sinking " << input << std::endl;
        });

        builder.make_edge(source_1_mod->output_port("source"), sink_1);

        auto source_2_mod = builder.make_module<TemplateModule<data_type_2_t>>("ModuleTemplateTest_mod2", config_2);

        auto sink_2 = builder.make_sink<data_type_2_t>("sink_2", [&packet_count_2](data_type_2_t input) {
            packet_count_2++;
            VLOG(10) << "Sinking " << input << std::endl;
        });

        builder.make_edge(source_2_mod->output_port("source"), sink_2);
    };

    m_pipeline->make_segment("SimpleModule_Segment", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(packet_count_1, 42);
    EXPECT_EQ(packet_count_2, 24);
}

#if !defined(__clang__) && defined(__GNUC__)
// Work around for GCC : https://gcc.gnu.org/bugzilla/show_bug.cgi?id=83258
auto F_1 = []() -> int { return 15; };
auto F_2 = []() -> std::string { return "test string"; };
#endif

TEST_F(SegmentTests, ModuleTemplateWithInitTest)
{
    using namespace modules;

    unsigned int packet_count_1{0};
    unsigned int packet_count_2{0};

    auto init_wrapper = [&packet_count_1, &packet_count_2](segment::Builder& builder) {
        using data_type_1_t = int;
        using data_type_2_t = std::string;

        auto config_1 = nlohmann::json();
        auto config_2 = nlohmann::json();

        unsigned int source_count_1{42};
        unsigned int source_count_2{24};

        config_1["source_count"] = source_count_1;
        config_2["source_count"] = source_count_2;

#if defined(__clang__)
        auto F_1 = []() -> int { return 15; };
        auto F_2 = []() -> std::string { return "test string"; };
#endif

        auto source_1_mod = builder.make_module<TemplateWithInitModule<data_type_1_t, F_1>>(
            "ModuleTemplateWithInitTest_mod1", config_1);

        auto sink_1 = builder.make_sink<data_type_1_t>("sink_1", [&packet_count_1](data_type_1_t input) {
            assert(input == 15);
            packet_count_1++;
            VLOG(10) << "Sinking " << input << std::endl;
        });

        builder.make_edge(source_1_mod->output_port("source"), sink_1);

        auto source_2_mod = builder.make_module<TemplateWithInitModule<data_type_2_t, F_2>>(
            "ModuleTemplateWithInitTest_mod2", config_2);

        auto sink_2 = builder.make_sink<data_type_2_t>("sink_2", [&packet_count_2](data_type_2_t input) {
            assert(input == "test string");
            packet_count_2++;
            VLOG(10) << "Sinking " << input << std::endl;
        });

        builder.make_edge(source_2_mod->output_port("source"), sink_2);
    };

    m_pipeline->make_segment("SimpleModule_Segment", init_wrapper);

    auto options = std::make_shared<Options>();
    options->topology().user_cpuset("0-1");
    options->topology().restrict_gpus(true);

    Executor executor(options);
    executor.register_pipeline(std::move(m_pipeline));
    executor.start();
    executor.join();

    EXPECT_EQ(packet_count_1, 42);
    EXPECT_EQ(packet_count_2, 24);
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

TEST_F(SegmentTests, DynamicModuleLoadTest)
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

TEST_F(SegmentTests, DynamicModuleRegistrationTest)
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

TEST_F(SegmentTests, DynamicModulePluginRegistrationTest)
{
    using namespace srf::modules;

    std::string module_path = get_modules_path() + "libdynamic_test_module.so";

    auto plugin = std::unique_ptr<ModulePluginLibrary>{};
    plugin      = ModulePluginLibrary::acquire(std::move(plugin), module_path);

    plugin->load();

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

    const char **module_list;
    auto module_count = plugin->list_modules(module_list);
    EXPECT_EQ(module_count, 3);

    //EXPECT_TRUE(std::equal(expected_modules.begin(), expected_modules.begin() + 3, actual_modules.begin()));

    plugin->unload();
    registered_modules = ModuleRegistry::registered_modules();

    EXPECT_TRUE(registered_modules.find("srf_unittest_cpp_dynamic") == registered_modules.end());
    EXPECT_TRUE(registered_modules.find("srf_unittest_cpp_dynamic_2") == registered_modules.end());
    EXPECT_TRUE(registered_modules.find("srf_unittest_cpp_dynamic_3") == registered_modules.end());
}

TEST_F(SegmentTests, DynamicModuleBadVersionTest)
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

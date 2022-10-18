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

#include "srf/experimental/modules/segment_modules.hpp"
#include "srf/segment/builder.hpp"

#include <typeinfo>

namespace srf::modules {

class SimpleModule : public SegmentModule
{
  public:
    SimpleModule(std::string module_name);
    SimpleModule(std::string module_name, nlohmann::json config);

    void initialize(segment::Builder& builder) override;

    bool m_was_configured{false};

  private:
    bool m_initialized{false};
};

SimpleModule::SimpleModule(std::string module_name) : SegmentModule(std::move(module_name)) {}

SimpleModule::SimpleModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

void SimpleModule::initialize(segment::Builder& builder)
{
    VLOG(10) << "MyModule::operator() called for '" << this->name() << "'" << std::endl;

    if (config().contains("simple_key_1"))
    {
        m_was_configured = true;
    }

    /** First linear path **/
    auto input1 = builder.make_node<bool, unsigned int>(this->get_module_component_name("input1"),
                                                        rxcpp::operators::map([this](bool input) {
                                                            unsigned int output = 42;
                                                            return output;
                                                        }));

    auto internal1 = builder.make_node<unsigned int, std::string>(
        this->get_module_component_name("_internal1_"), rxcpp::operators::map([this](unsigned int input) {
            auto output = std::to_string(input);
            VLOG(10) << "Created output1 << " << output << std::endl;
            return output;
        }));

    builder.make_edge(input1, internal1);

    auto output1 = builder.make_node<std::string, std::string>(
        this->get_module_component_name("output1"), rxcpp::operators::map([this](std::string input) { return input; }));

    builder.make_edge(internal1, output1);

    /** Second linear path **/
    auto input2 = builder.make_node<bool, unsigned int>(this->get_module_component_name("input2"),
                                                        rxcpp::operators::map([this](bool input) {
                                                            unsigned int output = 42;
                                                            return output;
                                                        }));

    auto internal2 = builder.make_node<unsigned int, std::string>(
        this->get_module_component_name("_internal2_"), rxcpp::operators::map([this](unsigned int input) {
            auto output = std::to_string(input);
            VLOG(10) << "Created output2: " << output << std::endl;
            return output;
        }));

    builder.make_edge(input2, internal2);

    auto output2 = builder.make_node<std::string, std::string>(
        this->get_module_component_name("output2"), rxcpp::operators::map([this](std::string input) { return input; }));

    builder.make_edge(internal2, output2);

    register_input_port("input1", input1, &typeid(bool));
    register_output_port("output1", output1, &typeid(std::string));

    register_input_port("input2", input2, &typeid(bool));
    register_output_port("output2", output2, &typeid(std::string));

    m_initialized = true;
}

class ConfigurableModule : public SegmentModule
{
  public:
    ConfigurableModule(std::string module_name);
    ConfigurableModule(std::string module_name, nlohmann::json config);

    void initialize(segment::Builder& builder) override;

    bool m_was_configured{false};

  private:
    bool m_initialized;
};

ConfigurableModule::ConfigurableModule(std::string module_name) : SegmentModule(std::move(module_name)) {}
ConfigurableModule::ConfigurableModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

void ConfigurableModule::initialize(segment::Builder& builder)
{
    VLOG(10) << "MyModule::operator() called for '" << this->name() << "'" << std::endl;

    if (config().contains("config_key_1"))
    {
        m_was_configured = true;
    }

    auto input1 = builder.make_node<bool, unsigned int>(this->get_module_component_name("configurable_input_a"),
                                                        rxcpp::operators::map([this](bool input) {
                                                            unsigned int output = 42;
                                                            return output;
                                                        }));

    auto internal1 = builder.make_node<unsigned int, std::string>(
        this->get_module_component_name("_internal1_"), rxcpp::operators::map([this](unsigned int input) {
            auto output = std::to_string(input);
            VLOG(10) << "Created output1: " << output << std::endl;
            return output;
        }));

    builder.make_edge(input1, internal1);

    auto output1 =
        builder.make_node<std::string, std::string>(this->get_module_component_name("configurable_output_x"),
                                                    rxcpp::operators::map([this](std::string input) { return input; }));

    builder.make_edge(internal1, output1);

    register_input_port("configurable_input_a", input1, &typeid(bool));
    register_output_port("configurable_output_x", output1, &typeid(std::string));

    m_initialized = true;
}

}  // namespace srf::modules
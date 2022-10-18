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

namespace srf::modules {

class SimpleModule : public SegmentModule
{
  public:
    SimpleModule(std::string module_name);
    SimpleModule(std::string module_name, nlohmann::json config);

    std::vector<std::string> input_ids() const override;
    std::vector<std::string> output_ids() const override;

    segment_module_port_map_t input_ports() override;
    SegmentModulePortT input_ports(const std::string& input_name) override;

    segment_module_port_map_t output_ports() override;
    SegmentModulePortT output_ports(const std::string& output_name) override;

    void initialize(segment::Builder& builder) override;

    bool m_was_configured{false};

  private:
    bool m_initialized{false};

    std::vector<std::string> m_inputs{"input1", "input2"};
    std::vector<std::string> m_outputs{};

    segment_module_port_map_t m_input_ports{};
    segment_module_port_map_t m_output_ports{};
};

SimpleModule::SimpleModule(std::string module_name) : SegmentModule(std::move(module_name)) {}

SimpleModule::SimpleModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

std::vector<std::string> SimpleModule::input_ids() const
{
    return m_inputs;
}

std::vector<std::string> SimpleModule::output_ids() const
{
    return m_outputs;
}

SegmentModule::segment_module_port_map_t SimpleModule::input_ports()
{
    return segment_module_port_map_t{};
}

SegmentModule::SegmentModulePortT SimpleModule::input_ports(const std::string& input_name)
{
    if (m_input_ports.find(input_name) != m_input_ports.end())
    {
        return m_input_ports[input_name];
    }

    std::stringstream sstream;

    sstream << "Invalid port name: " << input_name;

    throw std::invalid_argument(sstream.str());
}

SegmentModule::segment_module_port_map_t SimpleModule::output_ports()
{
    return m_output_ports;
}

SegmentModule::SegmentModulePortT SimpleModule::output_ports(const std::string& output_name)
{
    if (m_output_ports.find(output_name) != m_input_ports.end())
    {
        return m_output_ports[output_name];
    }

    std::stringstream sstream;

    sstream << "Invalid port name: " << output_name;

    throw std::invalid_argument(sstream.str());
}

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

    m_input_ports["input1"] = input1;

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

    m_output_ports["output1"] = output1;

    /** Second linear path **/
    auto input2 = builder.make_node<bool, unsigned int>(this->get_module_component_name("input2"),
                                                        rxcpp::operators::map([this](bool input) {
                                                            unsigned int output = 42;
                                                            return output;
                                                        }));

    m_input_ports["input2"] = input2;

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

    m_output_ports["output2"] = output2;

    m_initialized = true;
}

class ConfigurableModule : public SegmentModule
{
  public:
    ConfigurableModule(std::string module_name);
    ConfigurableModule(std::string module_name, nlohmann::json config);

    std::vector<std::string> input_ids() const override;
    std::vector<std::string> output_ids() const override;

    segment_module_port_map_t input_ports() override;
    SegmentModulePortT input_ports(const std::string& input_name) override;

    segment_module_port_map_t output_ports() override;
    SegmentModulePortT output_ports(const std::string& output_name) override;

    void initialize(segment::Builder& builder) override;

    bool m_was_configured{false};

  private:
    std::vector<std::string> m_inputs{"configurable_input_a"};
    std::vector<std::string> m_outputs{"configurable_output_x"};

    segment_module_port_map_t m_input_ports{};
    segment_module_port_map_t m_output_ports{};

    bool m_initialized;
};

ConfigurableModule::ConfigurableModule(std::string module_name) : SegmentModule(std::move(module_name)) {}
ConfigurableModule::ConfigurableModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

std::vector<std::string> ConfigurableModule::input_ids() const
{
    return m_inputs;
}

std::vector<std::string> ConfigurableModule::output_ids() const
{
    return m_outputs;
}

SegmentModule::segment_module_port_map_t ConfigurableModule::input_ports()
{
    return m_input_ports;
}

SegmentModule::SegmentModulePortT ConfigurableModule::input_ports(const std::string& input_name)
{
    if (m_input_ports.find(input_name) != m_input_ports.end())
    {
        return m_input_ports[input_name];
    }

    std::stringstream sstream;

    sstream << "Invalid port name: " << input_name;

    throw std::invalid_argument(sstream.str());
}

SegmentModule::segment_module_port_map_t ConfigurableModule::output_ports()
{
    return m_output_ports;
}

SegmentModule::SegmentModulePortT ConfigurableModule::output_ports(const std::string& output_name)
{
    if (m_output_ports.find(output_name) != m_input_ports.end())
    {
        return m_output_ports[output_name];
    }

    std::stringstream sstream;

    sstream << "Invalid port name: " << output_name;

    throw std::invalid_argument(sstream.str());
}

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

    m_input_ports["configurable_input_a"] = input1;

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

    m_output_ports["configurable_output_x"] = output1;

    m_initialized = true;
}

}  // namespace srf::modules
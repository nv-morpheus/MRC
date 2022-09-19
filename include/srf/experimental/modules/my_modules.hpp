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

#include "segment_modules.hpp"

#include "srf/segment/builder.hpp"


namespace srf::modules {

class MyModule : public SegmentModule
{
  public:
    MyModule(std::string module_name);

    const std::vector<std::string> inputs() const override;
    const std::vector<std::string> outputs() const override;

    segment_module_port_map_t input_ports() override;
    SegmentModulePortT input_ports(const std::string& input_name) override;

    segment_module_port_map_t output_ports() override;
    SegmentModulePortT output_ports(const std::string& output_name) override;

    void initialize(segment::Builder& builder) override;

  private:
    bool m_initialized{false};

    std::vector<std::string> m_inputs{"input1", "input2"};
    std::vector<std::string> m_outputs{};

    segment_module_port_map_t m_input_ports{};
    segment_module_port_map_t m_output_ports{};
};

MyModule::MyModule(std::string module_name) : SegmentModule(std::move(module_name)) {}

const std::vector<std::string> MyModule::inputs() const
{
    return m_inputs;
}

const std::vector<std::string> MyModule::outputs() const
{
    return m_outputs;
}

SegmentModule::segment_module_port_map_t MyModule::input_ports()
{
    return segment_module_port_map_t{};
}

SegmentModule::SegmentModulePortT MyModule::input_ports(const std::string& input_name)
{
    if (m_input_ports.find(input_name) != m_input_ports.end()){
        return m_input_ports[input_name];
    }

    std::stringstream sstream;

    sstream << "Invalid port name: " << input_name;

    throw std::invalid_argument(sstream.str());
}

SegmentModule::segment_module_port_map_t MyModule::output_ports()
{
    return m_output_ports;
}

SegmentModule::SegmentModulePortT MyModule::output_ports(const std::string& output_name)
{
    if (m_output_ports.find(output_name) != m_input_ports.end()){
        return m_output_ports[output_name];
    }

    std::stringstream sstream;

    sstream << "Invalid port name: " << output_name;

    throw std::invalid_argument(sstream.str());
}

void MyModule::initialize(segment::Builder& builder)
{
    std::cout << "MyModule::operator() called for '" << this->name() << "'" << std::endl;

    /** First linear path **/
    auto input1 = builder.make_node<bool, unsigned int>(this->component_prefix() + "input1",
                                                        rxcpp::operators::map([this](bool input) {
                                                            unsigned int output = 42;
                                                            return output;
                                                        }));

    m_input_ports["input1"] = input1;

    auto internal1 = builder.make_node<unsigned int, std::string>(
        this->component_prefix() + "_internal1_", rxcpp::operators::map([this](unsigned int input) {
            auto output = std::to_string(input);
            std::cout << "Created output1 << " << output << std::endl;
            return output;
        }));

    builder.make_edge(input1, internal1);

    auto output1 = builder.make_node<std::string, std::string>(
        this->component_prefix() + "output1", rxcpp::operators::map([this](std::string input) { return input; }));

    builder.make_edge(internal1, output1);

    m_output_ports["output1"] = output1;

    /** Second linear path **/
    auto input2 = builder.make_node<bool, unsigned int>(this->component_prefix() + "input2",
                                                        rxcpp::operators::map([this](bool input) {
                                                            unsigned int output = 42;
                                                            return output;
                                                        }));

    m_input_ports["input2"] = input2;

    auto internal2 = builder.make_node<unsigned int, std::string>(
        this->component_prefix() + "_internal2_", rxcpp::operators::map([this](unsigned int input) {
            auto output = std::to_string(input);
            std::cout << "Created output2: " << output << std::endl;
            return output;
        }));

    builder.make_edge(input2, internal2);

    auto output2 = builder.make_node<std::string, std::string>(
        this->component_prefix() + "output2", rxcpp::operators::map([this](std::string input) { return input; }));

    builder.make_edge(internal2, output2);

    m_output_ports["output2"] = output2;
}

class MyOtherModule : public SegmentModule
{
  public:
    MyOtherModule(const std::string& module_name);

    const std::vector<std::string> inputs() const override;
    const std::vector<std::string> outputs() const override;

    segment_module_port_map_t input_ports() override;
    SegmentModulePortT input_ports(const std::string& input_name) override;

    segment_module_port_map_t output_ports() override;
    SegmentModulePortT output_ports(const std::string& output_name) override;

    void initialize(segment::Builder& builder) override;

  private:
    std::vector<std::string> m_inputs{"other_input_a"};
    std::vector<std::string> m_outputs{"other_output_x"};

    segment_module_port_map_t m_input_ports{};
    segment_module_port_map_t m_output_ports{};
};

MyOtherModule::MyOtherModule(const std::string& module_name) : SegmentModule(module_name) {}

const std::vector<std::string> MyOtherModule::inputs() const
{
    return m_inputs;
}

const std::vector<std::string> MyOtherModule::outputs() const
{
    return m_outputs;
}

SegmentModule::segment_module_port_map_t MyOtherModule::input_ports()
{
    return m_input_ports;
}

SegmentModule::SegmentModulePortT MyOtherModule::input_ports(const std::string& input_name)
{
    if (m_input_ports.find(input_name) != m_input_ports.end()){
        return m_input_ports[input_name];
    }

    std::stringstream sstream;

    sstream << "Invalid port name: " << input_name;

    throw std::invalid_argument(sstream.str());
}

SegmentModule::segment_module_port_map_t MyOtherModule::output_ports()
{
    return m_output_ports;
}

SegmentModule::SegmentModulePortT MyOtherModule::output_ports(const std::string& output_name)
{
    if (m_output_ports.find(output_name) != m_input_ports.end()){
        return m_output_ports[output_name];
    }

    std::stringstream sstream;

    sstream << "Invalid port name: " << output_name;

    throw std::invalid_argument(sstream.str());
}

void MyOtherModule::initialize(segment::Builder& builder)
{
    std::cout << "MyModule::operator() called for '" << this->name() << "'" << std::endl;
}

}
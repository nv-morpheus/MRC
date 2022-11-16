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

#include "srf/modules/sample_modules.hpp"

#include "rxcpp/operators/rx-map.hpp"
#include "rxcpp/sources/rx-iterate.hpp"

#include "srf/channel/status.hpp"
#include "srf/core/utils.hpp"
#include "srf/modules/segment_modules.hpp"
#include "srf/node/rx_node.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/node/rx_source.hpp"
#include "srf/segment/object.hpp"

#include <boost/hana/if.hpp>
#include <glog/logging.h>

#include <memory>
#include <ostream>
#include <string_view>
#include <vector>

namespace srf::modules {
SimpleModule::SimpleModule(std::string module_name) : SegmentModule(std::move(module_name)) {}

SimpleModule::SimpleModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

void SimpleModule::initialize(segment::Builder& builder)
{
    if (config().contains("simple_key_1"))
    {
        m_was_configured = true;
    }

    /** First linear path **/
    auto input1 = builder.make_node<bool, unsigned int>("input1", rxcpp::operators::map([](bool input) {
                                                            unsigned int output = 42;
                                                            return output;
                                                        }));

    auto internal1 =
        builder.make_node<unsigned int, std::string>("_internal1_", rxcpp::operators::map([](unsigned int input) {
                                                         auto output = std::to_string(input);
                                                         VLOG(10) << "Created output1 << " << output << std::endl;
                                                         return output;
                                                     }));

    builder.make_edge(input1, internal1);

    auto output1 = builder.make_node<std::string, std::string>(
        "output1", rxcpp::operators::map([](std::string input) { return input; }));

    builder.make_edge(internal1, output1);

    /** Second linear path **/
    auto input2 = builder.make_node<bool, unsigned int>("input2", rxcpp::operators::map([](bool input) {
                                                            unsigned int output = 42;
                                                            return output;
                                                        }));

    auto internal2 =
        builder.make_node<unsigned int, std::string>("_internal2_", rxcpp::operators::map([](unsigned int input) {
                                                         auto output = std::to_string(input);
                                                         VLOG(10) << "Created output2: " << output << std::endl;
                                                         return output;
                                                     }));

    builder.make_edge(input2, internal2);

    auto output2 = builder.make_node<std::string, std::string>(
        "output2", rxcpp::operators::map([](std::string input) { return input; }));

    builder.make_edge(internal2, output2);

    register_input_port("input1", input1);
    register_output_port("output1", output1);

    register_input_port("input2", input2);
    register_output_port("output2", output2);

    m_initialized = true;
}

std::string SimpleModule::module_type_name() const
{
    return std::string(::srf::type_name<type_t>());
}

ConfigurableModule::ConfigurableModule(std::string module_name) : SegmentModule(std::move(module_name)) {}
ConfigurableModule::ConfigurableModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

void ConfigurableModule::initialize(segment::Builder& builder)
{
    if (config().contains("config_key_1"))
    {
        m_was_configured = true;
    }

    auto input1 = builder.make_node<bool, unsigned int>("configurable_input_a", rxcpp::operators::map([](bool input) {
                                                            unsigned int output = 42;
                                                            return output;
                                                        }));

    auto internal1 =
        builder.make_node<unsigned int, std::string>("_internal1_", rxcpp::operators::map([](unsigned int input) {
                                                         auto output = std::to_string(input);
                                                         VLOG(10) << "Created output1: " << output << std::endl;
                                                         return output;
                                                     }));

    builder.make_edge(input1, internal1);

    auto output1 = builder.make_node<std::string, std::string>(
        "configurable_output_x", rxcpp::operators::map([](std::string input) { return input; }));

    builder.make_edge(internal1, output1);

    register_input_port("configurable_input_a", input1);
    register_output_port("configurable_output_x", output1);

    m_initialized = true;
}

std::string ConfigurableModule::module_type_name() const
{
    return std::string(::srf::type_name<type_t>());
}

SourceModule::SourceModule(std::string module_name) : SegmentModule(std::move(module_name)) {}
SourceModule::SourceModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

void SourceModule::initialize(segment::Builder& builder)
{
    unsigned int count{1};

    if (config().contains("source_count"))
    {
        count = config()["source_count"];
    }

    auto source = builder.make_source<bool>("source", [count](rxcpp::subscriber<bool>& sub) {
        if (sub.is_subscribed())
        {
            for (unsigned int i = 0; i < count; ++i)
            {
                sub.on_next(true);
            }
        }

        sub.on_completed();
    });

    // Register the submodules output as one of this module's outputs
    register_output_port("source", source);
}

std::string SourceModule::module_type_name() const
{
    return std::string(::srf::type_name<type_t>());
}

SinkModule::SinkModule(std::string module_name) : SegmentModule(std::move(module_name)) {}
SinkModule::SinkModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

void SinkModule::initialize(segment::Builder& builder)
{
    auto sink = builder.make_sink<bool>("sink", [this](bool input) {
        m_packet_count++;
        VLOG(10) << "Sinking " << input << std::endl;
    });

    // Register the submodules output as one of this module's outputs
    register_input_port("sink", sink);
}

std::string SinkModule::module_type_name() const
{
    return std::string(::srf::type_name<type_t>());
}

NestedModule::NestedModule(std::string module_name) : SegmentModule(std::move(module_name)) {}
NestedModule::NestedModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

void NestedModule::initialize(segment::Builder& builder)
{
    auto configurable_mod = builder.make_module<ConfigurableModule>("NestedModule_submod2");

    auto config            = nlohmann::json();
    config["source_count"] = 4;

    // Create a data source and attach it to our submodule
    auto source1 = builder.make_module<SourceModule>("source", config);

    builder.make_dynamic_edge<bool, bool>(source1->output_port("source"),
                                          configurable_mod->input_port("configurable_input_a"));

    // Register the submodules output as one of this module's outputs
    register_output_port("nested_module_output", configurable_mod->output_port("configurable_output_x"));
}

std::string NestedModule::module_type_name() const
{
    return std::string(::srf::type_name<type_t>());
}
}  // namespace srf::modules
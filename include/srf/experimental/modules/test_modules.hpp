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
#include "srf/experimental/modules/test_modules.hpp" // TODO: included for testing
#include "srf/segment/builder.hpp"

#include <typeindex>

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

    register_input_port("input1", input1, input1->object().sink_type());
    register_output_port("output1", output1, output1->object().source_type());

    register_input_port("input2", input2, input2->object().sink_type());
    register_output_port("output2", output2, output2->object().source_type());

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

    register_input_port("configurable_input_a", input1, input1->object().sink_type());
    register_output_port("configurable_output_x", output1, output1->object().source_type());

    m_initialized = true;
}

class SourceModule : public SegmentModule
{
  public:
    SourceModule(std::string module_name);
    SourceModule(std::string module_name, nlohmann::json config);

    void initialize(segment::Builder& builder) override;

    bool m_was_configured{false};

  private:
    bool m_initialized;
};

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
    register_output_port("source", source, source->object().source_type());
}

class SinkModule : public SegmentModule
{
  public:
    SinkModule(std::string module_name);
    SinkModule(std::string module_name, nlohmann::json config);

    void initialize(segment::Builder& builder) override;

    bool m_was_configured{false};
    unsigned int m_packet_count{0};

  private:
    bool m_initialized;
};

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
    register_input_port("sink", sink, sink->object().sink_type());
}

class NestedModule : public SegmentModule
{
  public:
    NestedModule(std::string module_name);
    NestedModule(std::string module_name, nlohmann::json config);

    void initialize(segment::Builder& builder) override;

    std::string module_name() const override;

    bool m_was_configured{false};

  private:
    bool m_initialized;
};

NestedModule::NestedModule(std::string module_name) : SegmentModule(std::move(module_name)) {}
NestedModule::NestedModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

std::string NestedModule::module_name() const
{
    return "[nested_module]";
}

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
    register_output_port("nested_module_output",
                         configurable_mod->output_port("configurable_output_x"),
                         configurable_mod->output_port_type_id("configurable_output_x"));
}

template <typename OutputTypeT>
class TemplateModule : public SegmentModule
{
  public:
    TemplateModule(std::string module_name);
    TemplateModule(std::string module_name, nlohmann::json config);

    void initialize(segment::Builder& builder) override;

    std::string module_name() const override;

    bool m_was_configured{false};

  private:
    bool m_initialized;
};

template <typename OutputTypeT>
TemplateModule<OutputTypeT>::TemplateModule(std::string module_name) : SegmentModule(std::move(module_name))
{}

template <typename OutputTypeT>
TemplateModule<OutputTypeT>::TemplateModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

template <typename OutputTypeT>
std::string TemplateModule<OutputTypeT>::module_name() const
{
    return "[template_module]";
}

template <typename OutputTypeT>
void TemplateModule<OutputTypeT>::initialize(segment::Builder& builder)
{
    unsigned int count{1};

    if (config().contains("source_count"))
    {
        count = config()["source_count"];
    }

    auto source = builder.make_source<OutputTypeT>("source", [count](rxcpp::subscriber<OutputTypeT>& sub) {
        if (sub.is_subscribed())
        {
            for (unsigned int i = 0; i < count; ++i)
            {
                sub.on_next(std::move(OutputTypeT()));
            }
        }

        sub.on_completed();
    });

    // Register the submodules output as one of this module's outputs
    register_output_port("source", source, source->object().source_type());
}

template <typename OutputTypeT, OutputTypeT (*initializer)()>
class TemplateWithInitModule : public SegmentModule
{
  public:
    TemplateWithInitModule(std::string module_name);
    TemplateWithInitModule(std::string module_name, nlohmann::json config);

    void initialize(segment::Builder& builder) override;

    std::string module_name() const override;

    bool m_was_configured{false};

  private:
    bool m_initialized;
};

template <typename OutputTypeT, OutputTypeT (*Initializer)()>
TemplateWithInitModule<OutputTypeT, Initializer>::TemplateWithInitModule(std::string module_name) : SegmentModule(std::move(module_name))
{}

template <typename OutputTypeT, OutputTypeT (*Initializer)()>
TemplateWithInitModule<OutputTypeT, Initializer>::TemplateWithInitModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

template <typename OutputTypeT, OutputTypeT (*Initializer)()>
std::string TemplateWithInitModule<OutputTypeT, Initializer>::module_name() const
{
    return "[template_module]";
}

template <typename OutputTypeT, OutputTypeT (*Initializer)()>
void TemplateWithInitModule<OutputTypeT, Initializer>::initialize(segment::Builder& builder)
{
    unsigned int count{1};

    if (config().contains("source_count"))
    {
        count = config()["source_count"];
    }

    auto source = builder.make_source<OutputTypeT>("source", [count](rxcpp::subscriber<OutputTypeT>& sub) {
        if (sub.is_subscribed())
        {
            for (unsigned int i = 0; i < count; ++i)
            {
                auto data = Initializer();

                sub.on_next(std::move(data));
            }
        }

        sub.on_completed();
    });

    // Register the submodules output as one of this module's outputs
    register_output_port("source", source, source->object().source_type());
}


}  // namespace srf::modules

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

#include "srf/core/utils.hpp"
#include "srf/modules/segment_modules.hpp"
#include "srf/segment/builder.hpp"

#include <nlohmann/json.hpp>
#include <rxcpp/rx.hpp>

#include <string>
#include <utility>

namespace srf::modules {

/**
 * Create a 2 input 2 output SegmentModule
 * Inputs: input1:bool, input2:bool
 * Outputs: output1:std::string, output2:std::string
 */
class SimpleModule : public SegmentModule
{
    using type_t = SimpleModule;

  public:
    SimpleModule(std::string module_name);
    SimpleModule(std::string module_name, nlohmann::json config);

    bool m_was_configured{false};

  protected:
    void initialize(segment::Builder& builder) override;
    std::string module_type_name() const override;

  private:
    bool m_initialized{false};
};

/**
 * Create a 1 input 1 output module that sets 'm_was_configured' variable if 'config_key_1' is found in the config.
 * Inputs: configurable_input_a:bool
 * Outputs: configureable_output_x:std::string
 */
class ConfigurableModule : public SegmentModule
{
    using type_t = ConfigurableModule;

  public:
    ConfigurableModule(std::string module_name);
    ConfigurableModule(std::string module_name, nlohmann::json config);

    bool m_was_configured{false};

  protected:
    void initialize(segment::Builder& builder) override;
    std::string module_type_name() const override;

  private:
    bool m_initialized;
};

/**
 * Create a module that acts as a data source with one output
 * By default emits a single value, configurable by passing 'source_count' in the config.
 * Outputs: source:bool
 */
class SourceModule : public SegmentModule
{
    using type_t = SourceModule;

  public:
    SourceModule(std::string module_name);
    SourceModule(std::string module_name, nlohmann::json config);

  protected:
    void initialize(segment::Builder& builder) override;
    std::string module_type_name() const override;
};

/**
 * Create a module that acts as a data sink with one input
 * Inputs: sink:bool
 */
class SinkModule : public SegmentModule
{
    using type_t = SinkModule;

  public:
    SinkModule(std::string module_name);
    SinkModule(std::string module_name, nlohmann::json config);

    unsigned int m_packet_count{0};

  protected:
    void initialize(segment::Builder& builder) override;
    std::string module_type_name() const override;
};

/**
 * Creates a single output module that:
 *  - Creates a nested ConfigurableModule
 *  - Creates a nested SourceModule
 *  - Creates an edge between the SourceModule's output and the ConfigurableModule's input
 *  - Publishes the ConfigurableModule's 'configurable_output_x' as NestedModule's 'nested_module_output'
 *  Outputs: nested_module_output:bool
 */
class NestedModule : public SegmentModule
{
    using type_t = NestedModule;

  public:
    NestedModule(std::string module_name);
    NestedModule(std::string module_name, nlohmann::json config);

  protected:
    void initialize(segment::Builder& builder) override;
    std::string module_type_name() const override;
};

/**
 * Creates a data source that emits OutputTypeT data elements
 * @tparam OutputTypeT Type of data to emit
 * Outputs: source:OutputTypeT
 */
template <typename OutputTypeT>
class TemplateModule : public SegmentModule
{
    using type_t = TemplateModule<OutputTypeT>;

  public:
    TemplateModule(std::string module_name);
    TemplateModule(std::string module_name, nlohmann::json config);

    void initialize(segment::Builder& builder) override;
    std::string module_type_name() const override;
};

template <typename OutputTypeT>
TemplateModule<OutputTypeT>::TemplateModule(std::string module_name) : SegmentModule(std::move(module_name))
{}

template <typename OutputTypeT>
TemplateModule<OutputTypeT>::TemplateModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

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
    register_output_port("source", source);
}

template <typename OutputTypeT>
std::string TemplateModule<OutputTypeT>::module_type_name() const
{
    return std::string(::srf::type_name<type_t>());
}

/**
 * Creates a data source that emits OutputTypeT data elements, and takes a lambda function used to initialize the
 * emitted data element.
 * @tparam OutputTypeT Type of data to emit
 * @tparam Initializer Lambda function taking no inputs and returning a object of OutputTypeT
 * Outputs: source:OutputTypeT
 */
template <typename OutputTypeT, OutputTypeT (*Initializer)()>
class TemplateWithInitModule : public SegmentModule
{
    using type_t = TemplateWithInitModule<OutputTypeT, Initializer>;

  public:
    TemplateWithInitModule(std::string module_name);
    TemplateWithInitModule(std::string module_name, nlohmann::json config);

    void initialize(segment::Builder& builder) override;
    std::string module_type_name() const override;
};

template <typename OutputTypeT, OutputTypeT (*Initializer)()>
TemplateWithInitModule<OutputTypeT, Initializer>::TemplateWithInitModule(std::string module_name) :
  SegmentModule(std::move(module_name))
{}

template <typename OutputTypeT, OutputTypeT (*Initializer)()>
TemplateWithInitModule<OutputTypeT, Initializer>::TemplateWithInitModule(std::string module_name,
                                                                         nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

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
    register_output_port("source", source);
}

template <typename OutputTypeT, OutputTypeT (*Initializer)()>
std::string TemplateWithInitModule<OutputTypeT, Initializer>::module_type_name() const
{
    return std::string(::srf::type_name<type_t>());
}

}  // namespace srf::modules

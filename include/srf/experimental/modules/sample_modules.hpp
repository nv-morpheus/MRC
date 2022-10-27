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

#include <nlohmann/json.hpp>

#include <string>
#include <typeindex>

namespace srf::modules {

class SimpleModule : public SegmentModule
{
  public:
    SimpleModule(std::string module_name);
    SimpleModule(std::string module_name, nlohmann::json config);

    bool m_was_configured{false};

  protected:
    void initialize(segment::Builder& builder) override;

  private:
    bool m_initialized{false};
};

class ConfigurableModule : public SegmentModule
{
  public:
    ConfigurableModule(std::string module_name);
    ConfigurableModule(std::string module_name, nlohmann::json config);

    bool m_was_configured{false};

  protected:
    void initialize(segment::Builder& builder) override;

  private:
    bool m_initialized;
};

class SourceModule : public SegmentModule
{
  public:
    SourceModule(std::string module_name);
    SourceModule(std::string module_name, nlohmann::json config);

    bool m_was_configured{false};

  protected:
    void initialize(segment::Builder& builder) override;

  private:
    bool m_initialized;
};

class SinkModule : public SegmentModule
{
  public:
    SinkModule(std::string module_name);
    SinkModule(std::string module_name, nlohmann::json config);

    bool m_was_configured{false};
    unsigned int m_packet_count{0};

  protected:
    void initialize(segment::Builder& builder) override;

  private:
    bool m_initialized;
};

class NestedModule : public SegmentModule
{
  public:
    NestedModule(std::string module_name);
    NestedModule(std::string module_name, nlohmann::json config);

    std::string module_name() const override;

    bool m_was_configured{false};

  protected:
    void initialize(segment::Builder& builder) override;

  private:
    bool m_initialized;
};

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
TemplateWithInitModule<OutputTypeT, Initializer>::TemplateWithInitModule(std::string module_name) :
  SegmentModule(std::move(module_name))
{}

template <typename OutputTypeT, OutputTypeT (*Initializer)()>
TemplateWithInitModule<OutputTypeT, Initializer>::TemplateWithInitModule(std::string module_name,
                                                                         nlohmann::json config) :
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

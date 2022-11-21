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

#include "srf/channel/ingress.hpp"
#include "srf/exceptions/runtime_error.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/sink_properties.hpp"
#include "srf/node/source_properties.hpp"
#include "srf/node/type_traits.hpp"
#include "srf/runnable/launch_options.hpp"
#include "srf/runnable/runnable.hpp"
#include "srf/segment/forward.hpp"

#include <memory>
#include <string>
#include <type_traits>

namespace srf::segment {

struct ObjectProperties
{
    virtual ~ObjectProperties() = 0;

    virtual std::string name() const      = 0;
    virtual std::string type_name() const = 0;

    virtual bool is_sink() const   = 0;
    virtual bool is_source() const = 0;

    virtual node::SinkPropertiesBase& sink_base()     = 0;
    virtual node::SourcePropertiesBase& source_base() = 0;

    template <typename T>
    node::SinkProperties<T>& sink_typed();

    template <typename T>
    node::SourceProperties<T>& source_typed();

    virtual bool is_runnable() const = 0;

    virtual runnable::LaunchOptions& launch_options()             = 0;
    virtual const runnable::LaunchOptions& launch_options() const = 0;
};

inline ObjectProperties::~ObjectProperties() = default;

template <typename T>
node::SinkProperties<T>& ObjectProperties::sink_typed()
{
    auto& base = sink_base();
    auto* sink = dynamic_cast<node::SinkProperties<T>*>(&base);

    if (sink == nullptr)
    {
        LOG(ERROR) << "Failed to cast " << type_name() << " to "
                   << "SinkProperties<" << std::string(srf::type_name<T>()) << "> from "
                   << "SinkProperties<" << base.sink_type_name() << ">.";
        throw exceptions::SrfRuntimeError("Failed to cast Sink to requested SinkProperties<T>");
    }

    return *sink;
}

template <typename T>
node::SourceProperties<T>& ObjectProperties::source_typed()
{
    auto& base   = source_base();
    auto* source = dynamic_cast<node::SourceProperties<T>*>(&base);

    if (source == nullptr)
    {
        LOG(ERROR) << "Failed to cast " << type_name() << " to "
                   << "SourceProperties<" << std::string(srf::type_name<T>()) << "> from "
                   << "SourceProperties<" << base.source_type_name() << ">.";
        throw exceptions::SrfRuntimeError("Failed to cast Source to requested SourceProperties<T>");
    }

    return *source;
}

// Object

template <typename ObjectT>
class Object : public virtual ObjectProperties
{
  public:
    ObjectT& object();

    std::string name() const final;
    std::string type_name() const final;

    bool is_source() const final;
    bool is_sink() const final;

    node::SinkPropertiesBase& sink_base() final;
    node::SourcePropertiesBase& source_base() final;

    bool is_runnable() const final
    {
        return static_cast<bool>(std::is_base_of_v<runnable::Runnable, ObjectT>);
    }

    runnable::LaunchOptions& launch_options() final
    {
        if (!is_runnable())
        {
            LOG(ERROR) << "Segment Object is not Runnable; access to LaunchOption forbidden";
            throw exceptions::SrfRuntimeError("not a runnable");
        }
        return m_launch_options;
    }

    const runnable::LaunchOptions& launch_options() const final
    {
        if (!is_runnable())
        {
            LOG(ERROR) << "Segment Object is not Runnable; access to LaunchOption forbidden";
            throw exceptions::SrfRuntimeError("not a runnable");
        }
        return m_launch_options;
    }
  protected:
    void set_name(const std::string& name);

  private:
    std::string m_name{};

    virtual ObjectT* get_object() const = 0;
    runnable::LaunchOptions m_launch_options;
};

template <typename ObjectT>
ObjectT& Object<ObjectT>::object()
{
    auto* node = get_object();
    if (node == nullptr)
    {
        LOG(ERROR) << "Error accessing the Object API; Nodes are moved from the Segment API to the Executor "
                      "when the "
                      "pipeline is started.";
        throw exceptions::SrfRuntimeError("Object API is unavailable - expected if the Pipeline is running.");
    }
    return *node;
}

template <typename ObjectT>
void Object<ObjectT>::set_name(const std::string& name)
{
    m_name = name;
}

template <typename ObjectT>
std::string Object<ObjectT>::name() const
{
    return m_name;
}

template <typename ObjectT>
std::string Object<ObjectT>::type_name() const
{
    return std::string(::srf::type_name<ObjectT>());
}

template <typename ObjectT>
bool Object<ObjectT>::is_source() const
{
    return std::is_base_of_v<node::SourcePropertiesBase, ObjectT>;
}

template <typename ObjectT>
bool Object<ObjectT>::is_sink() const
{
    return std::is_base_of_v<node::SinkPropertiesBase, ObjectT>;
}

template <typename ObjectT>
node::SinkPropertiesBase& Object<ObjectT>::sink_base()
{
    if constexpr (!std::is_base_of_v<node::SinkPropertiesBase, ObjectT>)
    {
        LOG(ERROR) << type_name() << " is not a Sink";
        throw exceptions::SrfRuntimeError("Object is not a Sink");
    }

    auto* base = dynamic_cast<node::SinkPropertiesBase*>(get_object());
    CHECK(base);
    return *base;
}

template <typename ObjectT>
node::SourcePropertiesBase& Object<ObjectT>::source_base()
{
    if constexpr (!std::is_base_of_v<node::SourcePropertiesBase, ObjectT>)
    {
        LOG(ERROR) << type_name() << " is not a Source";
        throw exceptions::SrfRuntimeError("Object is not a Source");
    }

    auto* base = dynamic_cast<node::SourcePropertiesBase*>(get_object());
    CHECK(base);
    return *base;
}

}  // namespace srf::segment

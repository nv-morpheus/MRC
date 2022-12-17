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

#include "mrc/channel/ingress.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/channel_holder.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/node/type_traits.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/runnable.hpp"
#include "mrc/segment/forward.hpp"

#include <memory>
#include <string>
#include <type_traits>

namespace mrc::segment {

struct ObjectProperties
{
    virtual ~ObjectProperties() = 0;

    virtual std::string name() const      = 0;
    virtual std::string type_name() const = 0;

    virtual bool is_sink() const   = 0;
    virtual bool is_source() const = 0;

    virtual std::type_index sink_type(bool ignore_holder = false) const   = 0;
    virtual std::type_index source_type(bool ignore_holder = false) const = 0;

    virtual bool is_ingress_acceptor() const = 0;
    virtual bool is_ingress_provider() const = 0;
    virtual bool is_egress_acceptor() const  = 0;
    virtual bool is_egress_provider() const  = 0;

    // virtual node::SinkPropertiesBase& sink_base() = 0;
    // virtual node::SourcePropertiesBase& source_base() = 0;

    virtual node::IIngressAcceptorBase& ingress_acceptor_base() = 0;
    virtual node::IIngressProviderBase& ingress_provider_base() = 0;
    virtual node::IEgressAcceptorBase& egress_acceptor_base()   = 0;
    virtual node::IEgressProviderBase& egress_provider_base()   = 0;

    // template <typename T>
    // node::SinkProperties<T>& sink_typed();

    // template <typename T>
    // node::SourceProperties<T>& source_typed();

    template <typename T>
    node::IIngressAcceptor<T>& ingress_acceptor_typed();

    template <typename T>
    node::IEgressProvider<T>& egress_provider_typed();

    template <typename T>
    node::IIngressProvider<T>& ingress_provider_typed();

    template <typename T>
    node::IEgressAcceptor<T>& egress_acceptor_typed();

    virtual bool is_runnable() const = 0;

    virtual runnable::LaunchOptions& launch_options()             = 0;
    virtual const runnable::LaunchOptions& launch_options() const = 0;
};

inline ObjectProperties::~ObjectProperties() = default;

template <typename T>
node::IIngressAcceptor<T>& ObjectProperties::ingress_acceptor_typed()
{
    auto& base             = this->ingress_acceptor_base();
    auto* ingress_acceptor = dynamic_cast<node::IIngressAcceptor<T>*>(&base);

    if (ingress_acceptor == nullptr)
    {
        LOG(ERROR) << "Failed to cast " << type_name() << " to "
                   << "IIngressAcceptor<" << std::string(mrc::type_name<T>()) << ">"
                   << "IIngressAcceptor<" << ::mrc::type_name(base.ingress_acceptor_type().full_type()) << ">.";
        throw exceptions::MrcRuntimeError("Failed to cast Sink to requested IIngressAcceptor<T>");
    }

    return *ingress_acceptor;
}

template <typename T>
node::IIngressProvider<T>& ObjectProperties::ingress_provider_typed()
{
    auto& base             = this->ingress_provider_base();
    auto* ingress_provider = dynamic_cast<node::IIngressProvider<T>*>(&base);

    if (ingress_provider == nullptr)
    {
        LOG(ERROR) << "Failed to cast " << type_name() << " to "
                   << "IIngressProvider<" << std::string(mrc::type_name<T>()) << ">"
                   << "IIngressProvider<" << ::mrc::type_name(base.ingress_provider_type().full_type()) << ">.";
        throw exceptions::MrcRuntimeError("Failed to cast Sink to requested IIngressProvider<T>");
    }

    return *ingress_provider;
}

template <typename T>
node::IEgressAcceptor<T>& ObjectProperties::egress_acceptor_typed()
{
    auto& base            = this->egress_acceptor_base();
    auto* egress_acceptor = dynamic_cast<node::IEgressAcceptor<T>*>(&base);

    if (egress_acceptor == nullptr)
    {
        LOG(ERROR) << "Failed to cast " << type_name() << " to "
                   << "IEgressAcceptor<" << std::string(mrc::type_name<T>()) << ">"
                   << "IEgressAcceptor<" << ::mrc::type_name(base.egress_acceptor_type().full_type()) << ">.";
        throw exceptions::MrcRuntimeError("Failed to cast Sink to requested IEgressAcceptor<T>");
    }

    return *egress_acceptor;
}

template <typename T>
node::IEgressProvider<T>& ObjectProperties::egress_provider_typed()
{
    auto& base            = this->egress_provider_base();
    auto* egress_provider = dynamic_cast<node::IEgressProvider<T>*>(&base);

    if (egress_provider == nullptr)
    {
        LOG(ERROR) << "Failed to cast " << type_name() << " to "
                   << "IEgressProvider<" << std::string(mrc::type_name<T>()) << ">"
                   << "IEgressProvider<" << ::mrc::type_name(base.egress_provider_type().full_type()) << ">.";
        throw exceptions::MrcRuntimeError("Failed to cast Sink to requested IEgressProvider<T>");
    }

    return *egress_provider;
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

    std::type_index sink_type(bool ignore_holder) const final;
    std::type_index source_type(bool ignore_holder) const final;

    bool is_ingress_acceptor() const final;
    bool is_ingress_provider() const final;
    bool is_egress_acceptor() const final;
    bool is_egress_provider() const final;

    node::IIngressAcceptorBase& ingress_acceptor_base() final;
    node::IIngressProviderBase& ingress_provider_base() final;
    node::IEgressAcceptorBase& egress_acceptor_base() final;
    node::IEgressProviderBase& egress_provider_base() final;

    bool is_runnable() const final
    {
        return static_cast<bool>(std::is_base_of_v<runnable::Runnable, ObjectT>);
    }

    runnable::LaunchOptions& launch_options() final
    {
        if (!is_runnable())
        {
            LOG(ERROR) << "Segment Object is not Runnable; access to LaunchOption forbidden";
            throw exceptions::MrcRuntimeError("not a runnable");
        }
        return m_launch_options;
    }

    const runnable::LaunchOptions& launch_options() const final
    {
        if (!is_runnable())
        {
            LOG(ERROR) << "Segment Object is not Runnable; access to LaunchOption forbidden";
            throw exceptions::MrcRuntimeError("not a runnable");
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
        throw exceptions::MrcRuntimeError("Object API is unavailable - expected if the Pipeline is running.");
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
    return std::string(::mrc::type_name<ObjectT>());
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
std::type_index Object<ObjectT>::sink_type(bool ignore_holder) const
{
    CHECK(this->is_sink()) << "Object is not a sink";

    auto* base = dynamic_cast<node::SinkPropertiesBase*>(get_object());

    CHECK(base);

    return base->sink_type(ignore_holder);
}

template <typename ObjectT>
std::type_index Object<ObjectT>::source_type(bool ignore_holder) const
{
    CHECK(this->is_source()) << "Object is not a source";

    auto* base = dynamic_cast<node::SourcePropertiesBase*>(get_object());

    CHECK(base);

    return base->source_type(ignore_holder);
}

template <typename ObjectT>
bool Object<ObjectT>::is_ingress_acceptor() const
{
    return std::is_base_of_v<node::IIngressAcceptorBase, ObjectT>;
}

template <typename ObjectT>
bool Object<ObjectT>::is_ingress_provider() const
{
    return std::is_base_of_v<node::IIngressProviderBase, ObjectT>;
}

template <typename ObjectT>
bool Object<ObjectT>::is_egress_acceptor() const
{
    return std::is_base_of_v<node::IEgressAcceptorBase, ObjectT>;
}

template <typename ObjectT>
bool Object<ObjectT>::is_egress_provider() const
{
    return std::is_base_of_v<node::IEgressProviderBase, ObjectT>;
}

template <typename ObjectT>
node::IIngressAcceptorBase& Object<ObjectT>::ingress_acceptor_base()
{
    if constexpr (!std::is_base_of_v<node::IIngressAcceptorBase, ObjectT>)
    {
        LOG(ERROR) << type_name() << " is not a IIngressAcceptorBase";
        throw exceptions::MrcRuntimeError("Object is not a IIngressAcceptorBase");
    }

    auto* base = dynamic_cast<node::IIngressAcceptorBase*>(get_object());
    CHECK(base);
    return *base;
}

template <typename ObjectT>
node::IIngressProviderBase& Object<ObjectT>::ingress_provider_base()
{
    if constexpr (!std::is_base_of_v<node::IIngressProviderBase, ObjectT>)
    {
        LOG(ERROR) << type_name() << " is not a IIngressProviderBase";
        throw exceptions::MrcRuntimeError("Object is not a IIngressProviderBase");
    }

    auto* base = dynamic_cast<node::IIngressProviderBase*>(get_object());
    CHECK(base);
    return *base;
}

template <typename ObjectT>
node::IEgressAcceptorBase& Object<ObjectT>::egress_acceptor_base()
{
    if constexpr (!std::is_base_of_v<node::IEgressAcceptorBase, ObjectT>)
    {
        LOG(ERROR) << type_name() << " is not a IEgressAcceptorBase";
        throw exceptions::MrcRuntimeError("Object is not a IEgressAcceptorBase");
    }

    auto* base = dynamic_cast<node::IEgressAcceptorBase*>(get_object());
    CHECK(base);
    return *base;
}

template <typename ObjectT>
node::IEgressProviderBase& Object<ObjectT>::egress_provider_base()
{
    if constexpr (!std::is_base_of_v<node::IEgressProviderBase, ObjectT>)
    {
        LOG(ERROR) << type_name() << " is not a IEgressProviderBase";
        throw exceptions::MrcRuntimeError("Object is not a IEgressProviderBase");
    }

    auto* base = dynamic_cast<node::IEgressProviderBase*>(get_object());
    CHECK(base);
    return *base;
}

}  // namespace mrc::segment

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "mrc/edge/edge_builder.hpp"
#include "mrc/exceptions/runtime_error.hpp"
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

    virtual void set_name(const std::string& name) = 0;
    virtual std::string name() const               = 0;
    virtual std::string type_name() const          = 0;

    virtual bool is_sink() const   = 0;
    virtual bool is_source() const = 0;

    virtual std::type_index sink_type(bool ignore_holder = false) const   = 0;
    virtual std::type_index source_type(bool ignore_holder = false) const = 0;

    virtual bool is_writable_acceptor() const = 0;
    virtual bool is_writable_provider() const = 0;
    virtual bool is_readable_acceptor() const = 0;
    virtual bool is_readable_provider() const = 0;

    virtual edge::IWritableAcceptorBase& writable_acceptor_base() = 0;
    virtual edge::IWritableProviderBase& writable_provider_base() = 0;
    virtual edge::IReadableAcceptorBase& readable_acceptor_base() = 0;
    virtual edge::IReadableProviderBase& readable_provider_base() = 0;

    template <typename T>
    edge::IWritableAcceptor<T>& writable_acceptor_typed();

    template <typename T>
    edge::IReadableProvider<T>& readable_provider_typed();

    template <typename T>
    edge::IWritableProvider<T>& writable_provider_typed();

    template <typename T>
    edge::IReadableAcceptor<T>& readable_acceptor_typed();

    virtual bool is_runnable() const = 0;

    virtual runnable::LaunchOptions& launch_options()             = 0;
    virtual const runnable::LaunchOptions& launch_options() const = 0;

    virtual void destroy(){};
};

inline ObjectProperties::~ObjectProperties() = default;

template <typename T>
edge::IWritableAcceptor<T>& ObjectProperties::writable_acceptor_typed()
{
    auto& base              = this->writable_acceptor_base();
    auto* writable_acceptor = dynamic_cast<edge::IWritableAcceptor<T>*>(&base);

    if (writable_acceptor == nullptr)
    {
        LOG(ERROR) << "Failed to cast " << type_name() << " to "
                   << "IWritableAcceptor<" << std::string(mrc::type_name<T>()) << ">"
                   << "IWritableAcceptor<" << ::mrc::type_name(base.writable_acceptor_type().full_type()) << ">.";
        throw exceptions::MrcRuntimeError("Failed to cast Sink to requested IWritableAcceptor<T>");
    }

    return *writable_acceptor;
}

template <typename T>
edge::IWritableProvider<T>& ObjectProperties::writable_provider_typed()
{
    auto& base              = this->writable_provider_base();
    auto* writable_provider = dynamic_cast<edge::IWritableProvider<T>*>(&base);

    if (writable_provider == nullptr)
    {
        LOG(ERROR) << "Failed to cast " << type_name() << " to "
                   << "IWritableProvider<" << std::string(mrc::type_name<T>()) << ">"
                   << "IWritableProvider<" << ::mrc::type_name(base.writable_provider_type().full_type()) << ">.";
        throw exceptions::MrcRuntimeError("Failed to cast Sink to requested IWritableProvider<T>");
    }

    return *writable_provider;
}

template <typename T>
edge::IReadableAcceptor<T>& ObjectProperties::readable_acceptor_typed()
{
    auto& base              = this->readable_acceptor_base();
    auto* readable_acceptor = dynamic_cast<edge::IReadableAcceptor<T>*>(&base);

    if (readable_acceptor == nullptr)
    {
        LOG(ERROR) << "Failed to cast " << type_name() << " to "
                   << "IReadableAcceptor<" << std::string(mrc::type_name<T>()) << ">"
                   << "IReadableAcceptor<" << ::mrc::type_name(base.readable_acceptor_type().full_type()) << ">.";
        throw exceptions::MrcRuntimeError("Failed to cast Sink to requested IReadableAcceptor<T>");
    }

    return *readable_acceptor;
}

template <typename T>
edge::IReadableProvider<T>& ObjectProperties::readable_provider_typed()
{
    auto& base              = this->readable_provider_base();
    auto* readable_provider = dynamic_cast<edge::IReadableProvider<T>*>(&base);

    if (readable_provider == nullptr)
    {
        LOG(ERROR) << "Failed to cast " << type_name() << " to "
                   << "IReadableProvider<" << std::string(mrc::type_name<T>()) << ">"
                   << "IReadableProvider<" << ::mrc::type_name(base.readable_provider_type().full_type()) << ">.";
        throw exceptions::MrcRuntimeError("Failed to cast Sink to requested IReadableProvider<T>");
    }

    return *readable_provider;
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

    bool is_writable_acceptor() const final;
    bool is_writable_provider() const final;
    bool is_readable_acceptor() const final;
    bool is_readable_provider() const final;

    edge::IWritableAcceptorBase& writable_acceptor_base() final;
    edge::IWritableProviderBase& writable_provider_base() final;
    edge::IReadableAcceptorBase& readable_acceptor_base() final;
    edge::IReadableProviderBase& readable_provider_base() final;

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
    // Move to protected to allow only the IBuilder to set the name
    void set_name(const std::string& name) override;

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
                      "when the pipeline is started.";
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
bool Object<ObjectT>::is_writable_acceptor() const
{
    return std::is_base_of_v<edge::IWritableAcceptorBase, ObjectT>;
}

template <typename ObjectT>
bool Object<ObjectT>::is_writable_provider() const
{
    return std::is_base_of_v<edge::IWritableProviderBase, ObjectT>;
}

template <typename ObjectT>
bool Object<ObjectT>::is_readable_acceptor() const
{
    return std::is_base_of_v<edge::IReadableAcceptorBase, ObjectT>;
}

template <typename ObjectT>
bool Object<ObjectT>::is_readable_provider() const
{
    return std::is_base_of_v<edge::IReadableProviderBase, ObjectT>;
}

template <typename ObjectT>
edge::IWritableAcceptorBase& Object<ObjectT>::writable_acceptor_base()
{
    if constexpr (!std::is_base_of_v<edge::IWritableAcceptorBase, ObjectT>)
    {
        LOG(ERROR) << type_name() << " is not a IIngressAcceptorBase";
        throw exceptions::MrcRuntimeError("Object is not a IIngressAcceptorBase");
    }

    auto* base = dynamic_cast<edge::IWritableAcceptorBase*>(get_object());
    CHECK(base);
    return *base;
}

template <typename ObjectT>
edge::IWritableProviderBase& Object<ObjectT>::writable_provider_base()
{
    if constexpr (!std::is_base_of_v<edge::IWritableProviderBase, ObjectT>)
    {
        LOG(ERROR) << type_name() << " is not a IIngressProviderBase";
        throw exceptions::MrcRuntimeError("Object is not a IIngressProviderBase");
    }

    auto* base = dynamic_cast<edge::IWritableProviderBase*>(get_object());
    CHECK(base);
    return *base;
}

template <typename ObjectT>
edge::IReadableAcceptorBase& Object<ObjectT>::readable_acceptor_base()
{
    if constexpr (!std::is_base_of_v<edge::IReadableAcceptorBase, ObjectT>)
    {
        LOG(ERROR) << type_name() << " is not a IEgressAcceptorBase";
        throw exceptions::MrcRuntimeError("Object is not a IEgressAcceptorBase");
    }

    auto* base = dynamic_cast<edge::IReadableAcceptorBase*>(get_object());
    CHECK(base);
    return *base;
}

template <typename ObjectT>
edge::IReadableProviderBase& Object<ObjectT>::readable_provider_base()
{
    if constexpr (!std::is_base_of_v<edge::IReadableProviderBase, ObjectT>)
    {
        LOG(ERROR) << type_name() << " is not a IEgressProviderBase";
        throw exceptions::MrcRuntimeError("Object is not a IEgressProviderBase");
    }

    auto* base = dynamic_cast<edge::IReadableProviderBase*>(get_object());
    CHECK(base);
    return *base;
}
}  // namespace mrc::segment

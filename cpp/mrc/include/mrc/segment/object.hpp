/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "mrc/edge/edge_readable.hpp"
#include "mrc/edge/edge_writable.hpp"
#include "mrc/exceptions/checks.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/node_parent.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/node/type_traits.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/runnable.hpp"
#include "mrc/segment/forward.hpp"
#include "mrc/type_traits.hpp"
#include "mrc/utils/tuple_utils.hpp"

#include <concepts>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <typeindex>
#include <utility>

namespace mrc::segment {

template <typename ObjectT>
class SharedObject;

template <typename ObjectT>
class ReferencedObject;

struct ObjectPropertiesState
{
    const std::string type_name;

    const bool is_sink;
    const bool is_source;

    // std::optional<std::type_index> sink_type             = std::nullopt;
    // std::optional<std::type_index> sink_type_no_holder   = std::nullopt;
    // std::optional<std::type_index> source_type           = std::nullopt;
    // std::optional<std::type_index> source_type_no_holder = std::nullopt;

    const bool is_writable_acceptor;
    const bool is_writable_provider;
    const bool is_readable_acceptor;
    const bool is_readable_provider;

    const bool is_runnable;

    bool is_initialized() const
    {
        return m_is_initialized;
    }

    const std::string& name() const
    {
        return m_name;
    }

    IBuilder* owning_builder() const
    {
        return m_owning_builder;
    }

    void initialize(std::string name, IBuilder* owning_builder)
    {
        MRC_CHECK2(!m_is_initialized) << "Object '" << name << "' is already initialized.";

        m_name           = std::move(name);
        m_owning_builder = owning_builder;
        m_is_initialized = true;
    }

    template <typename ObjectT>
    static std::shared_ptr<ObjectPropertiesState> create()
    {
        auto state = std::shared_ptr<ObjectPropertiesState>(new ObjectPropertiesState(
            /*.type_name            = */ std::string(::mrc::type_name<ObjectT>()),
            /*.is_sink              = */ std::is_base_of_v<node::SinkPropertiesBase, ObjectT>,
            /*.is_source            = */ std::is_base_of_v<node::SourcePropertiesBase, ObjectT>,
            /*.is_writable_acceptor = */ std::is_base_of_v<edge::IWritableAcceptorBase, ObjectT>,
            /*.is_writable_provider = */ std::is_base_of_v<edge::IWritableProviderBase, ObjectT>,
            /*.is_readable_acceptor = */ std::is_base_of_v<edge::IReadableAcceptorBase, ObjectT>,
            /*.is_readable_provider = */ std::is_base_of_v<edge::IReadableProviderBase, ObjectT>,
            /*.is_runnable          = */ std::is_base_of_v<runnable::Runnable, ObjectT>));

        // state->type_name = std::string(::mrc::type_name<ObjectT>());

        // state->is_sink   = std::is_base_of_v<node::SinkPropertiesBase, ObjectT>;
        // state->is_source = std::is_base_of_v<node::SourcePropertiesBase, ObjectT>;

        // if constexpr (is_base_of_template_v<node::SinkProperties, ObjectT>)
        // {
        //     using sink_t = typename ObjectT::sink_type_t;

        //     state->sink_type_no_holder = deduce_type_index<sink_t>(true);
        //     state->sink_type           = deduce_type_index<sink_t>(false);
        // }
        // else
        // {
        //     CHECK(!state->is_sink) << "Object is a sink but does not have sink properties";
        // }

        // if constexpr (is_base_of_template_v<node::SourceProperties, ObjectT>)
        // {
        //     using source_t = typename ObjectT::source_type_t;

        //     state->source_type_no_holder = deduce_type_index<source_t>(true);
        //     state->source_type           = deduce_type_index<source_t>(false);
        // }
        // else
        // {
        //     CHECK(!state->is_source) << "Object is a source but does not have source properties";
        // }

        // state->is_writable_acceptor = std::is_base_of_v<edge::IWritableAcceptorBase, ObjectT>;
        // state->is_writable_provider = std::is_base_of_v<edge::IWritableProviderBase, ObjectT>;
        // state->is_readable_acceptor = std::is_base_of_v<edge::IReadableAcceptorBase, ObjectT>;
        // state->is_readable_provider = std::is_base_of_v<edge::IReadableProviderBase, ObjectT>;

        // state->is_runnable = std::is_base_of_v<runnable::Runnable, ObjectT>;

        return state;
    }

  private:
    ObjectPropertiesState(std::string type_name,
                          bool is_sink,
                          bool is_source,
                          bool is_writable_acceptor,
                          bool is_writable_provider,
                          bool is_readable_acceptor,
                          bool is_readable_provider,
                          bool is_runnable) :
      type_name(std::move(type_name)),
      is_sink(is_sink),
      is_source(is_source),
      is_writable_acceptor(is_writable_acceptor),
      is_writable_provider(is_writable_provider),
      is_readable_acceptor(is_readable_acceptor),
      is_readable_provider(is_readable_provider),
      is_runnable(is_runnable)
    {}

    // Will be set by the builder class when the object is added to a segment
    bool m_is_initialized{false};

    std::string m_name{""};

    // The owning builder. Once set, name cannot be changed
    IBuilder* m_owning_builder{nullptr};
};

class ObjectProperties
{
  public:
    virtual ~ObjectProperties() = default;

    void initialize(std::string name, IBuilder* owning_builder)
    {
        // Set our name first
        this->get_state().initialize(name, owning_builder);

        // Initialize the children
        this->init_children();
    }

    virtual std::string name() const
    {
        return this->get_state().name();
    }

    virtual std::string type_name() const
    {
        return this->get_state().type_name;
    }

    virtual bool is_sink() const
    {
        return this->get_state().is_sink;
    }

    virtual bool is_source() const
    {
        return this->get_state().is_source;
    }

    virtual std::type_index sink_type(bool ignore_holder = false) const = 0;

    virtual std::type_index source_type(bool ignore_holder = false) const = 0;

    bool is_writable_acceptor() const
    {
        return this->get_state().is_writable_acceptor;
    }
    bool is_writable_provider() const
    {
        return this->get_state().is_writable_provider;
    }
    bool is_readable_acceptor() const
    {
        return this->get_state().is_readable_acceptor;
    }
    bool is_readable_provider() const
    {
        return this->get_state().is_readable_provider;
    }

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

    virtual bool is_runnable() const
    {
        return this->get_state().is_runnable;
    }

    virtual IBuilder* owning_builder() const
    {
        return this->get_state().owning_builder();
    }

    virtual runnable::LaunchOptions& launch_options()             = 0;
    virtual const runnable::LaunchOptions& launch_options() const = 0;

    virtual bool has_child(const std::string& name) const                                        = 0;
    virtual std::shared_ptr<ObjectProperties> get_child(const std::string& name) const           = 0;
    virtual const std::map<std::string, std::shared_ptr<ObjectProperties>>& get_children() const = 0;

  protected:
    ObjectProperties() = default;

    virtual const ObjectPropertiesState& get_state() const = 0;
    virtual ObjectPropertiesState& get_state()             = 0;

  private:
    virtual void init_children() = 0;

    friend class IBuilder;
};

template <typename T>
edge::IWritableAcceptor<T>& ObjectProperties::writable_acceptor_typed()
{
    auto& base              = this->writable_acceptor_base();
    auto* writable_acceptor = dynamic_cast<edge::IWritableAcceptor<T>*>(&base);

    if (writable_acceptor == nullptr)
    {
        LOG(ERROR) << "Failed to cast " << type_name() << " to " << "IWritableAcceptor<"
                   << std::string(mrc::type_name<T>()) << ">" << "IWritableAcceptor<"
                   << ::mrc::type_name(base.writable_acceptor_type().full_type()) << ">.";
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
        LOG(ERROR) << "Failed to cast " << type_name() << " to " << "IWritableProvider<"
                   << std::string(mrc::type_name<T>()) << ">" << "IWritableProvider<"
                   << ::mrc::type_name(base.writable_provider_type().full_type()) << ">.";
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
        LOG(ERROR) << "Failed to cast " << type_name() << " to " << "IReadableAcceptor<"
                   << std::string(mrc::type_name<T>()) << ">" << "IReadableAcceptor<"
                   << ::mrc::type_name(base.readable_acceptor_type().full_type()) << ">.";
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
        LOG(ERROR) << "Failed to cast " << type_name() << " to " << "IReadableProvider<"
                   << std::string(mrc::type_name<T>()) << ">" << "IReadableProvider<"
                   << ::mrc::type_name(base.readable_provider_type().full_type()) << ">.";
        throw exceptions::MrcRuntimeError("Failed to cast Sink to requested IReadableProvider<T>");
    }

    return *readable_provider;
}

// template <typename T>
// std::type_index deduce_type_index(bool ignore_holder)
// {
//     if (ignore_holder)
//     {
//         if constexpr (is_smart_ptr_v<T>)
//         {
//             return std::type_index(typeid(typename T::element_type));
//         }
//     }

//     return std::type_index(typeid(T));
// }

// Object
template <typename ObjectT>
class Object : public virtual ObjectProperties, public std::enable_shared_from_this<Object<ObjectT>>
{
  public:
    // Object(const Object& other) : m_name(other.m_name), m_launch_options(other.m_launch_options) {}
    // Object(Object&&)                 = delete;
    // Object& operator=(const Object&) = delete;
    // Object& operator=(Object&&)      = delete;

    ObjectT& object();
    const ObjectT& object() const;

    // std::string name() const final;
    // std::string type_name() const final;

    // bool is_source() const final;
    // bool is_sink() const final;

    std::type_index sink_type(bool ignore_holder) const final;
    std::type_index source_type(bool ignore_holder) const final;

    // bool is_writable_acceptor() const final;
    // bool is_writable_provider() const final;
    // bool is_readable_acceptor() const final;
    // bool is_readable_provider() const final;

    edge::IWritableAcceptorBase& writable_acceptor_base() final;
    edge::IWritableProviderBase& writable_provider_base() final;
    edge::IReadableAcceptorBase& readable_acceptor_base() final;
    edge::IReadableProviderBase& readable_provider_base() final;

    // bool is_runnable() const final
    // {
    //     return static_cast<bool>(std::is_base_of_v<runnable::Runnable, ObjectT>);
    // }

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

    bool has_child(const std::string& name) const override
    {
        // First, split the name into the local and child names
        auto child_name_start_idx = name.find("/");

        if (child_name_start_idx != std::string::npos)
        {
            auto local_name = name.substr(0, child_name_start_idx);
            auto child_name = name.substr(child_name_start_idx + 1);

            // Check if the local name matches
            auto found = m_children.find(local_name);

            if (found == m_children.end())
            {
                return false;
            }

            // Now check if the child exists
            return found->second->has_child(child_name);
        }

        return m_children.contains(name);
    }

    std::shared_ptr<ObjectProperties> get_child(const std::string& name) const override
    {
        auto local_name = name;
        std::string child_name;

        // First, split the name into the local and child names
        auto child_name_start_idx = name.find("/");

        if (child_name_start_idx != std::string::npos)
        {
            local_name = name.substr(0, child_name_start_idx);
            child_name = name.substr(child_name_start_idx + 1);
        }

        auto found = m_children.find(local_name);

        if (found == m_children.end())
        {
            throw exceptions::MrcRuntimeError("Child " + local_name + " not found in " + this->name());
        }

        if (!child_name.empty())
        {
            return found->second->get_child(child_name);
        }

        return found->second;
    }

    const std::map<std::string, std::shared_ptr<ObjectProperties>>& get_children() const override
    {
        return m_children;
    }

    template <typename U>
        requires std::derived_from<ObjectT, U>
    std::shared_ptr<ReferencedObject<U>> as() const
    {
        auto shared_object = std::make_shared<ReferencedObject<U>>(*const_cast<Object*>(this));

        return shared_object;
    }

  protected:
    Object() : m_state(ObjectPropertiesState::create<ObjectT>())
    {
        LOG(INFO) << "Creating Object '" << this->name() << "' with type: " << this->type_name();
    }

    template <typename U>
        requires std::derived_from<U, ObjectT>
    Object(const Object<U>& other) :
      ObjectProperties(other),
      m_state(ObjectPropertiesState::create<U>()),
      m_launch_options(other.m_launch_options),
      m_children(other.m_children)
    {
        LOG(INFO) << "Copying Object '" << this->name() << "' from type: " << other.type_name()
                  << " to type: " << this->type_name();
    }

    const ObjectPropertiesState& get_state() const override
    {
        return *m_state;
    }

    ObjectPropertiesState& get_state() override
    {
        return *m_state;
    }

    // // Move to protected to allow only the IBuilder to set the name
    // void set_name(const std::string& name) override;

  private:
    virtual ObjectT* get_object() const = 0;

    void init_children() override
    {
        if constexpr (is_base_of_template<node::HomogeneousNodeParent, ObjectT>::value)
        {
            using child_node_t = typename ObjectT::child_node_t;

            // Get a map of the name/reference pairs from the NodeParent
            auto children_ref_pairs = this->object().get_children_refs();

            // Now loop and add any new children
            for (const auto& [name, child_ref] : children_ref_pairs)
            {
                auto child_obj = std::make_shared<SharedObject<child_node_t>>(this->shared_from_this(), child_ref);

                child_obj->initialize(name, this->owning_builder());

                m_children.emplace(name, std::move(child_obj));
            }
        }

        if constexpr (is_base_of_template<node::HeterogeneousNodeParent, ObjectT>::value)
        {
            using child_types_t = typename ObjectT::child_types_t;

            // Get the name/reference pairs from the NodeParent
            auto children_ref_pairs = this->object().get_children_refs();

            // Finally, convert the tuple of name/ChildObject pairs into a map
            utils::tuple_for_each(
                children_ref_pairs,
                [this]<typename ChildIndexT>(std::pair<std::string, std::reference_wrapper<ChildIndexT>>& pair,
                                             size_t idx) {
                    auto child_obj = std::make_shared<SharedObject<ChildIndexT>>(this->shared_from_this(), pair.second);

                    child_obj->initialize(pair.first, this->owning_builder());

                    m_children.emplace(pair.first, std::move(child_obj));
                });
        }
    }

    std::shared_ptr<ObjectPropertiesState> m_state;

    runnable::LaunchOptions m_launch_options;

    std::map<std::string, std::shared_ptr<ObjectProperties>> m_children;
    // std::map<std::string, std::function<std::shared_ptr<ObjectProperties>()>> m_create_children_fns;

    // Allows converting to base classes
    template <typename U>
    friend class Object;
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
const ObjectT& Object<ObjectT>::object() const
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

// template <typename ObjectT>
// void Object<ObjectT>::set_name(const std::string& name)
// {
//     m_name = name;
// }

// template <typename ObjectT>
// std::string Object<ObjectT>::name() const
// {
//     return m_name;
// }

// template <typename ObjectT>
// std::string Object<ObjectT>::type_name() const
// {
//     return std::string(::mrc::type_name<ObjectT>());
// }

// template <typename ObjectT>
// bool Object<ObjectT>::is_source() const
// {
//     return std::is_base_of_v<node::SourcePropertiesBase, ObjectT>;
// }

// template <typename ObjectT>
// bool Object<ObjectT>::is_sink() const
// {
//     return std::is_base_of_v<node::SinkPropertiesBase, ObjectT>;
// }

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

// template <typename ObjectT>
// bool Object<ObjectT>::is_writable_acceptor() const
// {
//     return std::is_base_of_v<edge::IWritableAcceptorBase, ObjectT>;
// }

// template <typename ObjectT>
// bool Object<ObjectT>::is_writable_provider() const
// {
//     return std::is_base_of_v<edge::IWritableProviderBase, ObjectT>;
// }

// template <typename ObjectT>
// bool Object<ObjectT>::is_readable_acceptor() const
// {
//     return std::is_base_of_v<edge::IReadableAcceptorBase, ObjectT>;
// }

// template <typename ObjectT>
// bool Object<ObjectT>::is_readable_provider() const
// {
//     return std::is_base_of_v<edge::IReadableProviderBase, ObjectT>;
// }

template <typename ObjectT>
edge::IWritableAcceptorBase& Object<ObjectT>::writable_acceptor_base()
{
    // if constexpr (!std::is_base_of_v<edge::IWritableAcceptorBase, ObjectT>)
    // {
    //     LOG(ERROR) << type_name() << " is not a IIngressAcceptorBase";
    //     throw exceptions::MrcRuntimeError("Object is not a IIngressAcceptorBase");
    // }

    auto* base = dynamic_cast<edge::IWritableAcceptorBase*>(get_object());
    CHECK(base) << type_name() << " is not a IIngressAcceptorBase";
    return *base;
}

template <typename ObjectT>
edge::IWritableProviderBase& Object<ObjectT>::writable_provider_base()
{
    // if constexpr (!std::is_base_of_v<edge::IWritableProviderBase, ObjectT>)
    // {
    //     LOG(ERROR) << type_name() << " is not a IIngressProviderBase";
    //     throw exceptions::MrcRuntimeError("Object is not a IIngressProviderBase");
    // }

    auto* base = dynamic_cast<edge::IWritableProviderBase*>(get_object());
    CHECK(base) << type_name() << " is not a IWritableProviderBase";
    return *base;
}

template <typename ObjectT>
edge::IReadableAcceptorBase& Object<ObjectT>::readable_acceptor_base()
{
    // if constexpr (!std::is_base_of_v<edge::IReadableAcceptorBase, ObjectT>)
    // {
    //     LOG(ERROR) << type_name() << " is not a IEgressAcceptorBase";
    //     throw exceptions::MrcRuntimeError("Object is not a IEgressAcceptorBase");
    // }

    auto* base = dynamic_cast<edge::IReadableAcceptorBase*>(get_object());
    CHECK(base) << type_name() << " is not a IReadableAcceptorBase";
    return *base;
}

template <typename ObjectT>
edge::IReadableProviderBase& Object<ObjectT>::readable_provider_base()
{
    // if constexpr (!std::is_base_of_v<edge::IReadableProviderBase, ObjectT>)
    // {
    //     LOG(ERROR) << type_name() << " is not a IEgressProviderBase";
    //     throw exceptions::MrcRuntimeError("Object is not a IEgressProviderBase");
    // }

    auto* base = dynamic_cast<edge::IReadableProviderBase*>(get_object());
    CHECK(base) << type_name() << " is not a IReadableProviderBase";
    return *base;
}

template <typename ObjectT>
class SharedObject final : public Object<ObjectT>
{
  public:
    SharedObject(std::shared_ptr<const ObjectProperties> owner, std::reference_wrapper<ObjectT> resource) :
      m_owner(std::move(owner)),
      m_resource(std::move(resource))
    {}
    ~SharedObject() final = default;

  private:
    ObjectT* get_object() const final
    {
        return &m_resource.get();
    }

    std::shared_ptr<const ObjectProperties> m_owner;
    std::reference_wrapper<ObjectT> m_resource;
};

template <typename ObjectT>
class ReferencedObject final : public Object<ObjectT>
{
  public:
    template <typename U>
        requires std::derived_from<U, ObjectT>
    ReferencedObject(Object<U>& other) :
      Object<ObjectT>(other),
      m_owner(other.shared_from_this()),
      m_resource(other.object())
    {}

    ~ReferencedObject() final = default;

  private:
    ObjectT* get_object() const final
    {
        return &m_resource.get();
    }

    std::shared_ptr<const ObjectProperties> m_owner;
    std::reference_wrapper<ObjectT> m_resource;
};

}  // namespace mrc::segment

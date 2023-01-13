/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/channel/channel.hpp"
#include "mrc/channel/egress.hpp"
#include "mrc/channel/ingress.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/type_traits.hpp"
#include "mrc/utils/string_utils.hpp"

#include <glog/logging.h>
#include <sys/types.h>

#include <cstddef>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

namespace mrc::node {

template <typename T>
class EdgeHolder;

template <typename KeyT, typename T>
class MultiEdgeHolder;

template <typename T>
class EdgeHandle;

class IngressHandleObj;
class EgressHandleObj;

class EdgeLifetime
{
  public:
    EdgeLifetime(std::function<void()> destruct_fn, bool is_armed = true) :
      m_destruct_fn(std::move(destruct_fn)),
      m_is_armed(is_armed)
    {}

    EdgeLifetime(const EdgeLifetime& other) = delete;
    EdgeLifetime(EdgeLifetime&& other)
    {
        std::swap(m_is_armed, other.m_is_armed);
        std::swap(m_destruct_fn, other.m_destruct_fn);
    }

    EdgeLifetime& operator=(const EdgeLifetime& other) = delete;
    EdgeLifetime& operator=(EdgeLifetime&& other) noexcept
    {
        std::swap(m_is_armed, other.m_is_armed);
        std::swap(m_destruct_fn, other.m_destruct_fn);

        return *this;
    }

    ~EdgeLifetime()
    {
        if (m_is_armed && m_destruct_fn)
        {
            m_destruct_fn();
        }
    }

    bool is_armed() const
    {
        return m_is_armed;
    }

    void arm()
    {
        m_is_armed = true;
    }

    void disarm()
    {
        m_is_armed = false;
    }

  private:
    bool m_is_armed{true};
    std::function<void()> m_destruct_fn;
};

class EdgeTag
{
  public:
    virtual ~EdgeTag()
    {
        for (auto& c : m_connectors)
        {
            c.disarm();
        }

        m_connectors.clear();
        m_disconnectors.clear();
    };

    bool is_connected() const
    {
        return m_is_connected;
    }

    void add_connector(std::function<void()>&& on_connect_fn)
    {
        this->add_connector(EdgeLifetime(std::move(on_connect_fn), true));
    }

    void add_connector(EdgeLifetime&& connector)
    {
        m_connectors.emplace_back(std::move(connector));
    }

    void add_disconnector(std::function<void()>&& on_disconnect_fn)
    {
        this->add_disconnector(EdgeLifetime(std::move(on_disconnect_fn), false));
    }

    void add_disconnector(EdgeLifetime&& disconnector)
    {
        LOG_IF(WARNING, disconnector.is_armed()) << "Adding armed disconnector to edge. This will fire even if edge is "
                                                    "never connected";

        m_disconnectors.emplace_back(std::move(disconnector));
    }

  protected:
    void connect()
    {
        m_is_connected = true;

        // Clear the connectors to execute them
        m_connectors.clear();

        // Arm all of the disconnectors now that we are connected
        for (auto& c : m_disconnectors)
        {
            c.arm();
        }

        // For all linked edges, call connect
        for (auto& linked_edge : m_linked_edges)
        {
            linked_edge->connect();
        }
    }

    void add_linked_edge(std::shared_ptr<EdgeTag> linked_edge)
    {
        if (m_is_connected)
        {
            linked_edge->connect();
        }

        m_linked_edges.emplace_back(std::move(linked_edge));
    }

  private:
    bool m_is_connected{false};
    std::vector<EdgeLifetime> m_connectors;
    std::vector<EdgeLifetime> m_disconnectors;
    std::vector<std::shared_ptr<EdgeTag>> m_linked_edges;

    // Friend any type of edge handle to allow calling connect
    template <typename>
    friend class EdgeHandle;
};

template <typename T>
class EdgeHandle : public virtual EdgeTag
{
  public:
    // Friend the holder classes which are required to setup connections
    template <typename>
    friend class EdgeHolder;
    template <typename, typename>
    friend class MultiEdgeHolder;
};

class EdgeTypePair
{
  public:
    EdgeTypePair(const EdgeTypePair& other) = default;

    std::type_index full_type() const
    {
        if (m_is_deferred)
        {
            throw std::runtime_error("Should not call full_type() for deferred types. Check is_deferred() first.");
        }

        return m_full_type.value();
    }

    std::type_index unwrapped_type() const
    {
        if (m_is_deferred)
        {
            throw std::runtime_error("Should not call unwrapped_type() for deferred types. Check is_deferred() first.");
        }

        return m_unwrapped_type.value();
    }

    bool is_deferred() const
    {
        return m_is_deferred;
    }

    bool operator==(const EdgeTypePair& other) const
    {
        return m_is_deferred == other.m_is_deferred && m_full_type == other.m_full_type &&
               m_unwrapped_type == other.m_unwrapped_type;
    }

    template <typename T>
    static EdgeTypePair create()
    {
        if constexpr (is_smart_ptr<T>::value)
        {
            return {typeid(T), typeid(typename T::element_type), false};
        }
        else
        {
            return {typeid(T), typeid(T), false};
        }
    }

    static EdgeTypePair create_deferred()
    {
        return {std::nullopt, std::nullopt, true};
    }

  private:
    EdgeTypePair(std::optional<std::type_index> full_type,
                 std::optional<std::type_index> unwrapped_type,
                 bool is_deferred) :
      m_full_type(full_type),
      m_unwrapped_type(unwrapped_type),
      m_is_deferred(is_deferred)
    {
        CHECK((m_is_deferred && !m_full_type.has_value() && !m_unwrapped_type.has_value()) ||
              (!m_is_deferred && m_full_type.has_value() && m_unwrapped_type.has_value()))
            << "Inconsistent deferred setting with concrete types";
    }

    std::optional<std::type_index> m_full_type;       // Includes any wrappers like shared_ptr
    std::optional<std::type_index> m_unwrapped_type;  // Excludes any wrappers like shared_ptr if they exist
    bool m_is_deferred{false};                        // Whether or not this type is deferred or concrete
};

class EdgeHandleObj
{
  public:
    const EdgeTypePair& get_type() const
    {
        return m_type;
    }

  protected:
    EdgeHandleObj(EdgeTypePair type_pair, std::shared_ptr<EdgeTag> edge_handle) :
      m_type(type_pair),
      m_handle(std::move(edge_handle))
    {}

    std::shared_ptr<EdgeTag> get_handle() const
    {
        return m_handle;
    }

    template <typename T>
    std::shared_ptr<T> get_handle_typed() const
    {
        return std::dynamic_pointer_cast<T>(m_handle);
    }

  private:
    EdgeTypePair m_type;

    std::shared_ptr<EdgeTag> m_handle{};

    // // Allow EdgeBuilder to access the internal edge
    // friend EdgeBuilder;

    // Allow ingress and egress derived objects to specialize
    friend IngressHandleObj;
    friend EgressHandleObj;

    // Add EdgeHandle to unpack the object before discarding
    template <typename>
    friend class EdgeHolder;
    template <typename, typename>
    friend class MultiEdgeHolder;
};

class IEdgeWritableBase : public virtual EdgeTag
{
  public:
    ~IEdgeWritableBase() override = default;

    virtual EdgeTypePair get_type() const = 0;
};

class IEdgeReadableBase : public virtual EdgeTag
{
  public:
    ~IEdgeReadableBase() override = default;

    virtual EdgeTypePair get_type() const = 0;
};

template <typename T>
class IEdgeWritable : public virtual EdgeHandle<T>, public virtual IEdgeWritableBase
{
  public:
    EdgeTypePair get_type() const override
    {
        return EdgeTypePair::create<T>();
    }

    virtual channel::Status await_write(T&& data) = 0;

    // If the above overload cannot be matched, copy by value and move into the await_write(T&&) overload. This is only
    // necessary for lvalues. The template parameters give it lower priority in overload resolution.
    template <typename TT = T, typename = std::enable_if_t<std::is_copy_constructible_v<TT>>>
    inline channel::Status await_write(T data)
    {
        return await_write(std::move(data));
    }
};

template <typename T>
class IEdgeReadable : public virtual EdgeHandle<T>, public IEdgeReadableBase
{
  public:
    EdgeTypePair get_type() const override
    {
        return EdgeTypePair::create<T>();
    }

    virtual channel::Status await_read(T& t) = 0;
};

template <typename InputT, typename OutputT = InputT>
class ConvertingEdgeWritableBase : public IEdgeWritable<InputT>
{
  public:
    using input_t  = InputT;
    using output_t = OutputT;

    ConvertingEdgeWritableBase(std::shared_ptr<IEdgeWritable<OutputT>> downstream) : m_downstream(downstream)
    {
        this->add_linked_edge(downstream);
    }

  protected:
    inline IEdgeWritable<OutputT>& downstream() const
    {
        return *m_downstream;
    }

  private:
    std::shared_ptr<IEdgeWritable<OutputT>> m_downstream{};
};

template <typename InputT, typename OutputT = InputT>
class ConvertingEdgeReadableBase : public IEdgeReadable<OutputT>
{
  public:
    using input_t  = InputT;
    using output_t = OutputT;

    ConvertingEdgeReadableBase(std::shared_ptr<IEdgeReadable<InputT>> upstream) : m_upstream(upstream)
    {
        this->add_linked_edge(upstream);
    }

  protected:
    inline IEdgeReadable<InputT>& upstream() const
    {
        return *m_upstream;
    }

  private:
    std::shared_ptr<IEdgeReadable<InputT>> m_upstream{};
};

template <typename InputT, typename OutputT = InputT, typename EnableT = void>
class ConvertingEdgeWritable;

template <typename InputT, typename OutputT>
class ConvertingEdgeWritable<InputT, OutputT, std::enable_if_t<std::is_convertible_v<InputT, OutputT>>>
  : public ConvertingEdgeWritableBase<InputT, OutputT>
{
  public:
    using base_t = ConvertingEdgeWritableBase<InputT, OutputT>;
    using typename base_t::input_t;
    using typename base_t::output_t;

    using base_t::base_t;

    channel::Status await_write(input_t&& data) override
    {
        return this->downstream().await_write(std::move(data));
    }
};

template <typename InputT, typename OutputT = InputT, typename EnableT = void>
class ConvertingEdgeReadable;

template <typename InputT, typename OutputT>
class ConvertingEdgeReadable<InputT, OutputT, std::enable_if_t<std::is_convertible_v<InputT, OutputT>>>
  : public ConvertingEdgeReadableBase<InputT, OutputT>
{
  public:
    using base_t = ConvertingEdgeReadableBase<InputT, OutputT>;
    using typename base_t::input_t;
    using typename base_t::output_t;

    using base_t::base_t;

    channel::Status await_read(OutputT& data) override
    {
        InputT source_data;
        auto ret_val = this->upstream().await_read(source_data);

        // Convert to the sink type
        data = std::move(source_data);

        return ret_val;
    }
};

template <typename InputT, typename OutputT>
class LambdaConvertingEdgeWritable : public ConvertingEdgeWritableBase<InputT, OutputT>
{
  public:
    using base_t = ConvertingEdgeWritableBase<InputT, OutputT>;
    using typename base_t::input_t;
    using typename base_t::output_t;
    using lambda_fn_t = std::function<output_t(input_t&&)>;

    LambdaConvertingEdgeWritable(lambda_fn_t lambda_fn, std::shared_ptr<IEdgeWritable<output_t>> downstream) :
      ConvertingEdgeWritableBase<input_t, output_t>(downstream),
      m_lambda_fn(std::move(lambda_fn))
    {}

    channel::Status await_write(input_t&& data) override
    {
        return this->downstream().await_write(m_lambda_fn(std::move(data)));
    }

  private:
    lambda_fn_t m_lambda_fn{};
};

template <typename InputT, typename OutputT>
class LambdaConvertingEdgeReadable : public ConvertingEdgeReadableBase<InputT, OutputT>
{
  public:
    using base_t = ConvertingEdgeReadableBase<InputT, OutputT>;
    using typename base_t::input_t;
    using typename base_t::output_t;
    using lambda_fn_t = std::function<output_t(input_t&&)>;

    LambdaConvertingEdgeReadable(lambda_fn_t lambda_fn, std::shared_ptr<IEdgeReadable<input_t>> upstream) :
      ConvertingEdgeReadableBase<input_t, output_t>(upstream),
      m_lambda_fn(std::move(lambda_fn))
    {}

    channel::Status await_read(output_t& data) override
    {
        input_t source_data;
        auto ret_val = this->upstream().await_read(source_data);

        // Convert to the sink type
        data = m_lambda_fn(std::move(source_data));

        return ret_val;
    }

  private:
    lambda_fn_t m_lambda_fn{};
};

// EdgeHolder keeps shared pointer of EdgeChannel alive and
template <typename T>
class EdgeHolder
{
  public:
    EdgeHolder() = default;
    virtual ~EdgeHolder()
    {
        // Drop any edge connections before this object goes out of scope. This should execute any disconnectors
        m_connected_edge.reset();

        if (this->check_active_connection(false))
        {
            LOG(FATAL) << "A node was destructed which still had dependent connections. Nodes must be kept alive while "
                          "dependent connections are still active";
        }
    }

  protected:
    bool check_active_connection(bool do_throw = true) const
    {
        // Alive connection exists when the lock is true, lifetime is false or a connction object has been set
        if (m_owned_edge.lock() && !m_owned_edge_lifetime)
        {
            // Then someone is using this edge already, cant be changed
            if (do_throw)
            {
                throw std::runtime_error("Cant change edge after a connection has been made");
            }
            return true;
        }

        // Check for set connections. Must be connected to throw error
        if (m_connected_edge && m_connected_edge->is_connected())
        {
            // Then someone is using this edge already, cant be changed
            if (do_throw)
            {
                throw std::runtime_error(
                    "Cannot make multiple connections to the same node. Use dedicated Broadcast node");
            }
            return true;
        }

        return false;
    }

    void init_owned_edge(std::shared_ptr<EdgeHandle<T>> edge)
    {
        // Check for active connections
        this->check_active_connection();

        // Register a connector to run when edge is connected
        edge->add_connector([this]() {
            // Drop the object keeping the weak_edge alive
            this->m_owned_edge_lifetime.reset();
        });

        // Now register a disconnector to keep clean everything up. Only runs if connected
        edge->add_disconnector([this]() {
            this->m_owned_edge_lifetime.reset();
            this->m_owned_edge.reset();
        });

        // Set to the temp edge to ensure its alive until get_edge is called
        m_owned_edge_lifetime = edge;

        // Set to the weak ptr as well
        m_owned_edge = edge;
    }

    void init_connected_edge(std::shared_ptr<EdgeHandle<T>> edge)
    {
        // Check for active connections
        this->check_active_connection();

        m_connected_edge = edge;
    }

    std::shared_ptr<EdgeHandleObj> get_edge_connection() const
    {
        if (auto edge = m_owned_edge.lock())
        {
            return std::shared_ptr<EdgeHandleObj>(new EdgeHandleObj(EdgeTypePair::create<T>(), edge));
        }

        throw std::runtime_error("Must set an edge before calling get_edge");
    }

    void make_edge_connection(std::shared_ptr<EdgeHandleObj> edge_obj)
    {
        CHECK(edge_obj->get_type() == EdgeTypePair::create<T>()) << "Incoming edge connection is not the correct type. "
                                                                    "Make sure to call "
                                                                    "`EdgeBuilder::adapt_ingress<T>(edge)` or "
                                                                    "`EdgeBuilder::adapt_egress<T>(edge)` before "
                                                                    "calling "
                                                                    "make_edge_connection";

        // Unpack the edge, convert, and call the inner set_edge
        auto unpacked_edge = edge_obj->get_handle_typed<EdgeHandle<T>>();

        this->set_edge_handle(unpacked_edge);
    }

    void release_edge_connection()
    {
        m_owned_edge_lifetime.reset();
        m_connected_edge.reset();
    }

    const std::shared_ptr<EdgeHandle<T>>& get_connected_edge() const
    {
        return m_connected_edge;
    }

  private:
    void set_edge_handle(std::shared_ptr<EdgeHandle<T>> edge)
    {
        // Check for active connections
        this->check_active_connection();

        // Set to the temp edge to ensure its alive until get_edge is called
        m_connected_edge = edge;

        // Reset the weak_ptr since we dont own this edge
        m_owned_edge.reset();

        // Remove any init lifetime
        m_owned_edge_lifetime.reset();

        // Now indicate that we have a connection
        edge->connect();
    }

    // Used for retrieving the current edge without altering its lifetime
    std::weak_ptr<EdgeHandle<T>> m_owned_edge;

    // This object ensures that any initialized edge is kept alive and is cleared on connection
    std::shared_ptr<EdgeHandle<T>> m_owned_edge_lifetime;

    // Holds a pointer to any set edge (different from init edge). Maintains lifetime
    std::shared_ptr<EdgeHandle<T>> m_connected_edge;

    // Allow edge builder to call set_edge
    friend EdgeBuilder;

    // Allow multi edge holder to access protected elements
    template <typename KeyT, typename OtherT>
    friend class MultiEdgeHolder;
};

template <typename KeyT, typename T>
class MultiEdgeHolder
{
  public:
    MultiEdgeHolder()          = default;
    virtual ~MultiEdgeHolder() = default;

  protected:
    void init_owned_edge(KeyT key, std::shared_ptr<EdgeHandle<T>> edge)
    {
        auto& edge_pair = this->get_edge_pair(key, true);

        edge_pair.init_owned_edge(std::move(edge));
    }

    std::shared_ptr<EdgeHandleObj> get_edge_connection(const KeyT& key) const
    {
        auto& edge_pair = this->get_edge_pair(key);

        return edge_pair.get_edge_connection();
    }

    void make_edge_connection(KeyT key, std::shared_ptr<EdgeHandleObj> edge_obj)
    {
        auto& edge_pair = this->get_edge_pair(key, true);

        edge_pair.make_edge_connection(std::move(edge_obj));
    }

    void release_edge_connection(const KeyT& key)
    {
        auto& edge_pair = this->get_edge_pair(key, true);

        edge_pair.release_edge_connection();

        m_edges.erase(key);
    }

    const std::shared_ptr<EdgeHandle<T>>& get_connected_edge(const KeyT& key) const
    {
        auto& edge_pair = this->get_edge_pair(key);

        return edge_pair.get_connected_edge();
    }

    void release_edge_connections()
    {
        for (auto& [key, edge_pair] : m_edges)
        {
            edge_pair.release_edge_connection();
        }

        m_edges.clear();
    }

    size_t edge_connection_count() const
    {
        return m_edges.size();
    }

    std::vector<KeyT> edge_connection_keys() const
    {
        std::vector<KeyT> keys;

        for (const auto& [key, _] : m_edges)
        {
            keys.push_back(key);
        }

        return keys;
    }

    EdgeHolder<T>& get_edge_pair(KeyT key, bool create_if_missing = false)
    {
        auto found = m_edges.find(key);

        if (found == m_edges.end())
        {
            if (create_if_missing)
            {
                m_edges[key] = EdgeHolder<T>();
                return m_edges[key];
            }

            throw std::runtime_error(MRC_CONCAT_STR("Could not find edge pair for key: " << key));
        }

        return found->second;
    }

    EdgeHolder<T>& get_edge_pair(KeyT key)
    {
        auto found = m_edges.find(key);

        if (found == m_edges.end())
        {
            throw std::runtime_error(MRC_CONCAT_STR("Could not find edge pair for key: " << key));
        }

        return found->second;
    }

    const EdgeHolder<T>& get_edge_pair(KeyT key) const
    {
        auto found = m_edges.find(key);

        if (found == m_edges.end())
        {
            throw std::runtime_error(MRC_CONCAT_STR("Could not find edge pair for key: " << key));
        }

        return found->second;
    }

  private:
    void set_edge_handle(KeyT key, std::shared_ptr<EdgeHandle<T>> edge)
    {
        auto& edge_pair = this->get_edge_pair(key, true);

        edge_pair.set_edge_handle(std::move(edge));
    }

    // Keeps pairs of get_edge/set_edge for each key
    std::map<KeyT, EdgeHolder<T>> m_edges;

    // Allow edge builder to call set_edge
    friend EdgeBuilder;
};

class DeferredIngressHandleObj;

class IngressHandleObj : public EdgeHandleObj
{
  public:
    IngressHandleObj(std::shared_ptr<IEdgeWritableBase> ingress) : IngressHandleObj(ingress->get_type(), ingress) {}

    static std::shared_ptr<IngressHandleObj> from_typeless(std::shared_ptr<EdgeHandleObj> other)
    {
        auto typed_ingress = other->get_handle_typed<IEdgeWritableBase>();

        CHECK(typed_ingress) << "Could not convert to ingress";

        return std::make_shared<IngressHandleObj>(std::move(typed_ingress));
    }

    virtual bool is_deferred() const
    {
        return false;
    }

  protected:
    // Allow manually specifying the edge type
    IngressHandleObj(EdgeTypePair edge_type, std::shared_ptr<IEdgeWritableBase> ingress) :
      EdgeHandleObj(edge_type, ingress)
    {}

  private:
    std::shared_ptr<IEdgeWritableBase> get_ingress() const
    {
        return std::dynamic_pointer_cast<IEdgeWritableBase>(this->get_handle());
    }

    template <typename T>
    std::shared_ptr<IEdgeWritable<T>> get_ingress_typed() const
    {
        return std::dynamic_pointer_cast<IEdgeWritable<T>>(this->get_handle());
    }

    void set_ingress_handle(std::shared_ptr<IEdgeWritableBase> ingress)
    {
        this->m_type   = ingress->get_type();
        this->m_handle = ingress;
    }

    // Allow EdgeBuilder to unpack the edge
    friend EdgeBuilder;

    // Add deferred ingresses to set their deferred type
    friend DeferredIngressHandleObj;
};

class EgressHandleObj : public EdgeHandleObj
{
  public:
    EgressHandleObj(std::shared_ptr<IEdgeReadableBase> egress) : EdgeHandleObj(egress->get_type(), egress) {}

    static std::shared_ptr<EgressHandleObj> from_typeless(std::shared_ptr<EdgeHandleObj> other)
    {
        auto typed_ingress = other->get_handle_typed<IEdgeReadableBase>();

        CHECK(typed_ingress) << "Could not convert to egress";

        return std::make_shared<EgressHandleObj>(std::move(typed_ingress));
    }

  private:
    std::shared_ptr<IEdgeReadableBase> get_egress() const
    {
        return std::dynamic_pointer_cast<IEdgeReadableBase>(this->get_handle());
    }

    template <typename T>
    std::shared_ptr<IEdgeReadable<T>> get_egress_typed() const
    {
        return std::dynamic_pointer_cast<IEdgeReadable<T>>(this->get_handle());
    }

    void set_egress_handle(std::shared_ptr<IEdgeReadableBase> egress)
    {
        this->m_type   = egress->get_type();
        this->m_handle = egress;
    }

    friend EdgeBuilder;
};

class IEgressProviderBase
{
  public:
    virtual std::shared_ptr<EdgeTag> get_egress_typeless() const = 0;

    virtual std::shared_ptr<EgressHandleObj> get_egress_obj() const = 0;

    virtual EdgeTypePair egress_provider_type() const = 0;
};

class IEgressAcceptorBase
{
  public:
    virtual void set_egress_typeless(std::shared_ptr<EdgeTag> egress) = 0;

    virtual void set_egress_obj(std::shared_ptr<EgressHandleObj> egress) = 0;

    virtual EdgeTypePair egress_acceptor_type() const = 0;
};

class IIngressProviderBase
{
  public:
    virtual std::shared_ptr<EdgeTag> get_ingress_typeless() const = 0;

    virtual std::shared_ptr<IngressHandleObj> get_ingress_obj() const = 0;

    virtual EdgeTypePair ingress_provider_type() const = 0;
};

class IIngressAcceptorBase
{
  public:
    virtual void set_ingress_typeless(std::shared_ptr<EdgeTag> ingress) = 0;

    virtual void set_ingress_obj(std::shared_ptr<IngressHandleObj> ingress) = 0;

    virtual EdgeTypePair ingress_acceptor_type() const = 0;
};

template <typename KeyT>
class IMultiIngressAcceptorBase
{
  public:
    // virtual void set_ingress_typeless(std::shared_ptr<EdgeTag> ingress) = 0;

    virtual void set_ingress_obj(KeyT key, std::shared_ptr<IngressHandleObj> ingress) = 0;

    // virtual EdgeTypePair ingress_acceptor_type() const = 0;
};

template <typename T>
class IEgressProvider : public IEgressProviderBase
{
  public:
    // virtual std::shared_ptr<IEdgeReadable<T>> get_egress() const = 0;

    std::shared_ptr<EdgeTag> get_egress_typeless() const override
    {
        return nullptr;
        // return this->get_egress();
    }

    EdgeTypePair egress_provider_type() const override
    {
        return EdgeTypePair::create<T>();
    }

    // std::shared_ptr<EgressHandleObj> get_egress_obj() const override
    // {
    //     return std::make_shared<EgressHandleObj>(this->get_egress());
    // }
};

template <typename T>
class IEgressAcceptor : public IEgressAcceptorBase
{
  public:
    // virtual void set_egress(std::shared_ptr<IEdgeReadable<T>> egress) = 0;

    void set_egress_typeless(std::shared_ptr<EdgeTag> egress) override
    {
        // this->set_egress(std::dynamic_pointer_cast<IEdgeReadable<T>>(egress));
    }

    EdgeTypePair egress_acceptor_type() const override
    {
        return EdgeTypePair::create<T>();
    }
};

template <typename T>
class IIngressProvider : public IIngressProviderBase
{
  public:
    std::shared_ptr<EdgeTag> get_ingress_typeless() const override
    {
        // return std::dynamic_pointer_cast<EdgeTag>(this->get_ingress());
        return nullptr;
    }

    EdgeTypePair ingress_provider_type() const override
    {
        return EdgeTypePair::create<T>();
    }

    // std::shared_ptr<IngressHandleObj> get_ingress_obj() const override
    // {
    //     return std::make_shared<IngressHandleObj>(this->get_ingress());
    // }

    //   private:
    //     virtual std::shared_ptr<IEdgeWritable<T>> get_ingress() const = 0;
};

template <typename T>
class IIngressAcceptor : public IIngressAcceptorBase
{
  public:
    void set_ingress_typeless(std::shared_ptr<EdgeTag> ingress) override
    {
        // this->set_ingress(std::dynamic_pointer_cast<IEdgeWritable<T>>(ingress));
    }

    EdgeTypePair ingress_acceptor_type() const override
    {
        return EdgeTypePair::create<T>();
    }

    //   private:
    //     virtual void set_ingress(std::shared_ptr<IEdgeWritable<T>> ingress) = 0;
};

template <typename T, typename KeyT>
class IMultiIngressAcceptor : public IMultiIngressAcceptorBase<KeyT>
{};

}  // namespace mrc::node

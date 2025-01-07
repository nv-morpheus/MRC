/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/channel/status.hpp"  // IWYU pragma: export
#include "mrc/edge/edge_builder.hpp"
#include "mrc/edge/edge_readable.hpp"
#include "mrc/edge/edge_writable.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/type_traits.hpp"
#include "mrc/utils/type_utils.hpp"

#include <memory>
#include <stdexcept>
#include <typeindex>

namespace mrc::node {

template <typename T>
class NullReadableEdge : public edge::IEdgeReadable<T>
{
  public:
    virtual ~NullReadableEdge() = default;

    channel::Status await_read(T& t) override
    {
        throw std::runtime_error("Attempting to read from a null edge. Ensure an edge was established for all sinks.");

        return channel::Status::error;
    }

    channel::Status await_read_until(T& t, const mrc::channel::time_point_t& tp) override
    {
        throw std::runtime_error("Attempting to read from a null edge. Ensure an edge was established for all sinks.");

        return channel::Status::error;
    }
};

/**
 * @brief Type erased base class for the formation of all edges to a sink
 */
class SinkPropertiesBase
{
  public:
    virtual ~SinkPropertiesBase() = 0;

    /**
     * @brief std::type_index for sink type
     *
     * @param ignore_holder
     * @return std::type_index
     */
    virtual std::type_index sink_type(bool ignore_holder = false) const = 0;

    /**
     * @brief returns the name of the sink type
     *
     * @return std::string
     */
    virtual std::string sink_type_name() const = 0;

    /**
     * @brief UINT64 hash of sink type
     * @return std::size_t
     */
    std::size_t sink_type_hash() const
    {
        return sink_type().hash_code();
    }

  protected:
    SinkPropertiesBase() = default;
};

inline SinkPropertiesBase::~SinkPropertiesBase() = default;

/**
 * @brief Typed SinkProperties provides default implementations dependent only on the type T.
 */
template <typename T>
class SinkProperties : public edge::EdgeHolder<T>, public SinkPropertiesBase
{
  public:
    using sink_type_t = T;

    std::type_index sink_type(bool ignore_holder = false) const final
    {
        if (ignore_holder)
        {
            if constexpr (is_smart_ptr<T>::value)
            {
                return typeid(typename T::element_type);
            }
        }
        return typeid(T);
    }

    std::string sink_type_name() const final
    {
        return std::string(type_name<T>());
    }

  protected:
    SinkProperties()
    {
        // Set the default edge to be a null one in case no connection is made
        this->init_connected_edge(std::make_shared<NullReadableEdge<T>>());
    }

    std::shared_ptr<edge::IEdgeReadable<T>> get_readable_edge() const
    {
        return std::dynamic_pointer_cast<edge::IEdgeReadable<T>>(this->get_connected_edge());
    }
};

template <typename KeyT, typename T>
class MultiSinkProperties : public edge::MultiEdgeHolder<KeyT, T>, public SinkPropertiesBase
{
  public:
    using sink_type_t = T;

    std::type_index sink_type(bool ignore_holder = false) const final
    {
        if (ignore_holder)
        {
            if constexpr (is_smart_ptr<T>::value)
            {
                return typeid(typename T::element_type);
            }
        }
        return typeid(T);
    }

    std::string sink_type_name() const final
    {
        return std::string(type_name<T>());
    }

  protected:
    std::shared_ptr<edge::IEdgeReadable<T>> get_readable_edge(KeyT edge_key) const
    {
        return std::dynamic_pointer_cast<edge::IEdgeReadable<T>>(this->get_connected_edge(edge_key));
    }
};

template <typename T>
class ReadableAcceptor : public virtual SinkProperties<T>, public edge::IReadableAcceptor<T>
{
  public:
    ReadableAcceptor& operator=(ReadableAcceptor&& other)
    {
        // Only call concrete class
        SinkProperties<T>::operator=(std::move(other));
    }

  protected:
    void set_readable_edge_handle(std::shared_ptr<edge::ReadableEdgeHandle> egress) override
    {
        // Do any conversion to the correct type here
        auto adapted_egress = edge::EdgeBuilder::adapt_readable_edge<T>(egress);

        SinkProperties<T>::make_edge_connection(adapted_egress);
    }
};

template <typename T>
class WritableProvider : public virtual SinkProperties<T>, public edge::IWritableProvider<T>
{
  public:
    WritableProvider& operator=(WritableProvider&& other)
    {
        // Only call concrete class
        SinkProperties<T>::operator=(std::move(other));
    }

  protected:
    std::shared_ptr<edge::WritableEdgeHandle> get_writable_edge_handle() const override
    {
        return edge::WritableEdgeHandle::from_typeless(SinkProperties<T>::get_edge_connection());
    }
};

template <typename T>
class ReadableWritableSink : public WritableProvider<T>, public ReadableAcceptor<T>
{};

template <typename KeyT, typename T>
class MultiReadableAcceptor : public virtual MultiSinkProperties<KeyT, T>, public edge::IMultiReadableAcceptor<KeyT, T>
{
  public:
  protected:
    bool has_readable_edge(const KeyT& key) const override
    {
        return MultiSinkProperties<KeyT, T>::has_edge_connection(key);
    }

    void release_readable_edge(const KeyT& key) override
    {
        return MultiSinkProperties<KeyT, T>::release_edge_connection(key);
    }

    void release_readable_edges() override
    {
        return MultiSinkProperties<KeyT, T>::release_edge_connections();
    }

    size_t readable_edge_count() const override
    {
        return MultiSinkProperties<KeyT, T>::edge_connection_count();
    }

    std::vector<KeyT> readable_edge_keys() const override
    {
        return MultiSinkProperties<KeyT, T>::edge_connection_keys();
    }

    void set_readable_edge_handle(KeyT key, std::shared_ptr<edge::WritableEdgeHandle> egress) override
    {
        auto adapted_egress = edge::EdgeBuilder::adapt_readable_edge<T>(egress);
        MultiSinkProperties<KeyT, T>::make_edge_connection(key, adapted_egress);
    }
};

template <typename KeyT, typename T>
class MultiWritableProvider : public virtual MultiSinkProperties<KeyT, T>, public edge::IMultiWritableProvider<KeyT, T>
{
  public:
  protected:
    bool has_writable_edge(const KeyT& key) const override
    {
        return MultiSinkProperties<KeyT, T>::has_edge_connection(key);
    }

    void release_writable_edge(const KeyT& key) override
    {
        return MultiSinkProperties<KeyT, T>::release_edge_connection(key);
    }

    void release_writable_edges() override
    {
        return MultiSinkProperties<KeyT, T>::release_edge_connections();
    }

    size_t writable_edge_count() const override
    {
        return MultiSinkProperties<KeyT, T>::edge_connection_count();
    }

    std::vector<KeyT> writable_edge_keys() const override
    {
        return MultiSinkProperties<KeyT, T>::edge_connection_keys();
    }

    std::shared_ptr<edge::WritableEdgeHandle> get_writable_edge_handle(KeyT key) const override
    {
        return edge::WritableEdgeHandle::from_typeless(MultiSinkProperties<KeyT, T>::get_edge_connection(key));
    }
};

template <typename T>
class ForwardingWritableProvider : public WritableProvider<T>
{
  protected:
    class ForwardingEdge : public edge::IEdgeWritable<T>
    {
      public:
        ForwardingEdge(ForwardingWritableProvider<T>& parent) : m_parent(parent) {}

        ~ForwardingEdge() = default;

        channel::Status await_write(T&& t) override
        {
            return m_parent.on_next(std::move(t));
        }

      private:
        ForwardingWritableProvider<T>& m_parent;
    };

    ForwardingWritableProvider()
    {
        auto inner_edge = std::make_shared<ForwardingEdge>(*this);

        inner_edge->add_disconnector([this]() {
            // Only call the on_complete if we have been connected
            this->on_complete();
        });

        WritableProvider<T>::init_owned_edge(inner_edge);
    }

    virtual channel::Status on_next(T&& t) = 0;

    virtual void on_complete() {}
};

}  // namespace mrc::node

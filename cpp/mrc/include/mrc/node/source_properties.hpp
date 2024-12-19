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
#include "mrc/channel/status.hpp"  // IWYU pragma: export
#include "mrc/edge/edge_builder.hpp"
#include "mrc/edge/edge_readable.hpp"
#include "mrc/edge/edge_writable.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/type_traits.hpp"
#include "mrc/utils/type_utils.hpp"

#include <memory>
#include <string>
#include <typeindex>

namespace mrc::node {

template <typename T>
class NullWritableEdge : public edge::IEdgeWritable<T>
{
  public:
    virtual ~NullWritableEdge() = default;

    channel::Status await_write(T&& t) override
    {
        // Move to a new object and then let it go out of scope
        T dummy = std::move(t);

        return channel::Status::success;
    }
};

/**
 * @brief Type erased base class for the formation of all edges to a source
 */
class SourcePropertiesBase
{
  public:
    virtual ~SourcePropertiesBase() = 0;

    /**
     * @brief std::type_index for source type
     *
     * @param ignore_holder
     * @return std::type_index
     */
    virtual std::type_index source_type(bool ignore_holder = false) const = 0;

    /**
     * @brief returns the name of the source type
     *
     * @return std::string
     */
    virtual std::string source_type_name() const = 0;

    /**
     * @brief UINT64 hash of source type
     * @return std::size_t
     */
    std::size_t source_type_hash() const
    {
        return source_type().hash_code();
    }

  protected:
    SourcePropertiesBase() = default;
};

inline SourcePropertiesBase::~SourcePropertiesBase() = default;

/**
 * @brief Typed SourceProperties provides default implementations dependent only on the type T.
 */
template <typename T>
class SourceProperties : public edge::EdgeHolder<T>, public SourcePropertiesBase
{
  public:
    using source_type_t = T;

    std::type_index source_type(bool ignore_holder = false) const final
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

    std::string source_type_name() const final
    {
        return std::string(type_name<T>());
    }

  protected:
    SourceProperties()
    {
        // Set the default edge to be a null one in case no connection is made
        this->init_connected_edge(std::make_shared<NullWritableEdge<T>>());
    }

    std::shared_ptr<edge::IEdgeWritable<T>> get_writable_edge() const
    {
        return std::dynamic_pointer_cast<edge::IEdgeWritable<T>>(this->get_connected_edge());
    }
};

template <typename KeyT, typename T>
class MultiSourceProperties : public edge::MultiEdgeHolder<KeyT, T>, public SourcePropertiesBase
{
  public:
    using source_type_t = T;

    std::type_index source_type(bool ignore_holder = false) const final
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

    std::string source_type_name() const final
    {
        return std::string(type_name<T>());
    }

  protected:
    std::shared_ptr<edge::IEdgeWritable<T>> get_writable_edge(KeyT edge_key) const
    {
        return std::dynamic_pointer_cast<edge::IEdgeWritable<T>>(this->get_connected_edge(edge_key));
    }
};

template <typename T>
class ReadableProvider : public virtual SourceProperties<T>, public edge::IReadableProvider<T>
{
  public:
    ReadableProvider& operator=(ReadableProvider&& other)
    {
        // Only call concrete class
        SourceProperties<T>::operator=(std::move(other));
    }

  protected:
    std::shared_ptr<edge::ReadableEdgeHandle> get_readable_edge_handle() const override
    {
        return edge::ReadableEdgeHandle::from_typeless(SourceProperties<T>::get_edge_connection());
    }
};

template <typename T>
class WritableAcceptor : public virtual SourceProperties<T>, public edge::IWritableAcceptor<T>
{
  public:
    WritableAcceptor& operator=(WritableAcceptor&& other)
    {
        // Only call concrete class
        SourceProperties<T>::operator=(std::move(other));
    }

  protected:
    void set_writable_edge_handle(std::shared_ptr<edge::WritableEdgeHandle> ingress) override
    {
        // Do any conversion to the correct type here
        auto adapted_ingress = edge::EdgeBuilder::adapt_writable_edge<T>(ingress);

        SourceProperties<T>::make_edge_connection(adapted_ingress);
    }
};

template <typename T>
class ReadableWritableSource : public ReadableProvider<T>, public WritableAcceptor<T>
{};

template <typename KeyT, typename T>
class MultiReadableProvider : public virtual MultiSourceProperties<KeyT, T>,
                              public edge::IMultiReadableProvider<KeyT, T>
{
  public:
  protected:
    bool has_readable_edge(const KeyT& key) const override
    {
        return MultiSourceProperties<KeyT, T>::has_edge_connection(key);
    }

    void release_readable_edge(const KeyT& key) override
    {
        return MultiSourceProperties<KeyT, T>::release_edge_connection(key);
    }

    void release_readable_edges() override
    {
        return MultiSourceProperties<KeyT, T>::release_edge_connections();
    }

    size_t readable_edge_count() const override
    {
        return MultiSourceProperties<KeyT, T>::edge_connection_count();
    }

    std::vector<KeyT> readable_edge_keys() const override
    {
        return MultiSourceProperties<KeyT, T>::edge_connection_keys();
    }

    std::shared_ptr<edge::ReadableEdgeHandle> get_readable_edge_handle(KeyT key) const override
    {
        return edge::ReadableEdgeHandle::from_typeless(MultiSourceProperties<KeyT, T>::get_edge_connection(key));
    }
};

template <typename KeyT, typename T>
class MultiWritableAcceptor : public virtual MultiSourceProperties<KeyT, T>,
                              public edge::IMultiWritableAcceptor<KeyT, T>
{
  public:
  protected:
    bool has_writable_edge(const KeyT& key) const override
    {
        return MultiSourceProperties<KeyT, T>::has_edge_connection(key);
    }

    void release_writable_edge(const KeyT& key) override
    {
        return MultiSourceProperties<KeyT, T>::release_edge_connection(key);
    }

    void release_writable_edges() override
    {
        return MultiSourceProperties<KeyT, T>::release_edge_connections();
    }

    size_t writable_edge_count() const override
    {
        return MultiSourceProperties<KeyT, T>::edge_connection_count();
    }

    std::vector<KeyT> writable_edge_keys() const override
    {
        return MultiSourceProperties<KeyT, T>::edge_connection_keys();
    }

    void set_writable_edge_handle(KeyT key, std::shared_ptr<edge::WritableEdgeHandle> ingress) override
    {
        // Do any conversion to the correct type here
        auto adapted_ingress = edge::EdgeBuilder::adapt_writable_edge<T>(ingress);

        MultiSourceProperties<KeyT, T>::make_edge_connection(key, adapted_ingress);
    }
};

template <typename T>
class ForwardingReadableProvider : public ReadableProvider<T>
{
  protected:
    class ForwardingEdge : public edge::IEdgeReadable<T>
    {
      public:
        ForwardingEdge(ForwardingReadableProvider<T>& parent) : m_parent(parent) {}

        ~ForwardingEdge() = default;

        channel::Status await_read(T& t) override
        {
            return m_parent.get_next(t);
        }

        channel::Status await_read_until(T& t, const mrc::channel::time_point_t& tp) override
        {
            throw std::runtime_error("Not implemented");
        }

      private:
        ForwardingReadableProvider<T>& m_parent;
    };

    ForwardingReadableProvider()
    {
        auto inner_edge = std::make_shared<ForwardingEdge>(*this);

        inner_edge->add_disconnector([this]() {
            // Only call the on_complete if we have been connected
            this->on_complete();
        });

        ReadableProvider<T>::init_owned_edge(inner_edge);
    }

    virtual channel::Status get_next(T& t) = 0;

    virtual void on_complete() {}
};

}  // namespace mrc::node

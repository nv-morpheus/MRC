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
#include "mrc/edge/edge_writable.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/type_traits.hpp"
#include "mrc/types.hpp"  // for Mutex
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

  private:
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

  private:
    void set_writable_edge_handle(std::shared_ptr<edge::WritableEdgeHandle> ingress) override
    {
        // Do any conversion to the correct type here
        auto adapted_ingress = edge::EdgeBuilder::adapt_writable_edge<T>(ingress);

        SourceProperties<T>::make_edge_connection(adapted_ingress);
    }
};

template <typename T, typename KeyT>
class MultiIngressAcceptor : public virtual MultiSourceProperties<T, KeyT>, public edge::IMultiWritableAcceptor<T, KeyT>
{
  public:
  private:
    void set_writable_edge_handle(KeyT key, std::shared_ptr<edge::WritableEdgeHandle> ingress) override
    {
        // Do any conversion to the correct type here
        auto adapted_ingress = edge::EdgeBuilder::adapt_writable_edge<T>(ingress);

        MultiSourceProperties<T, KeyT>::make_edge_connection(key, adapted_ingress);
    }
};

template <typename T>
class ForwardingEgressProvider : public ReadableProvider<T>
{
  protected:
    struct State
    {
        Mutex m_mutex;
        bool m_is_destroyed{false};
    };

    class ForwardingEdge : public edge::IEdgeReadable<T>
    {
      public:
        ForwardingEdge(ForwardingEgressProvider<T>& parent, std::shared_ptr<State> state) :
          m_parent(parent),
          m_state(std::move(state))
        {}

        ~ForwardingEdge() = default;

        channel::Status await_read(T& t) override
        {
            std::lock_guard<decltype(m_state->m_mutex)> lock(m_state->m_mutex);
            if (!(m_state->m_is_destroyed))
            {
                return m_parent.get_next(t);
            }

            return channel::Status::closed;
        }

      private:
        ForwardingEgressProvider<T>& m_parent;
        std::shared_ptr<State> m_state;
    };

    ForwardingEgressProvider() : m_state(std::make_shared<State>())
    {
        auto inner_edge = std::make_shared<ForwardingEdge>(*this, m_state);

        inner_edge->add_disconnector([this, state = m_state]() {
            std::lock_guard<decltype(state->m_mutex)> lock(state->m_mutex);
            if (!(state->m_is_destroyed))
            {
                // Only call the on_complete if we have been connected and `this` is still alive
                this->on_complete();
            }
        });

        ReadableProvider<T>::init_owned_edge(inner_edge);
    }

    ~ForwardingEgressProvider()
    {
        SourceProperties<T>::disconnect();
        {
            std::lock_guard<decltype(m_state->m_mutex)> lock(m_state->m_mutex);
            m_state->m_is_destroyed = true;
        }
    }

    virtual channel::Status get_next(T& t) = 0;

    virtual void on_complete() {}

    std::shared_ptr<State> m_state;
};

}  // namespace mrc::node

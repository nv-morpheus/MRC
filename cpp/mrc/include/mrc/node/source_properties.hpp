/**
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
#include "mrc/node/channel_holder.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/type_traits.hpp"
#include "mrc/utils/type_utils.hpp"

#include <memory>
#include <string>
#include <typeindex>

namespace mrc::node {

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

  private:
    // /**
    //  * @brief Interface to the Channel Writer to accept an Edge from the Builder
    //  */
    // virtual void complete_edge(std::shared_ptr<channel::IngressHandle>) = 0;

    // needs to be able to call channel_ingress_for_sink
    friend EdgeBuilder;
};

inline SourcePropertiesBase::~SourcePropertiesBase() = default;

/**
 * @brief Typed SourceProperties provides default implementations dependent only on the type T.
 */
template <typename T>
class SourceProperties : public EdgeHolder<T>, public SourcePropertiesBase
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
    std::shared_ptr<IEdgeWritable<T>> get_writable_edge() const
    {
        return std::dynamic_pointer_cast<IEdgeWritable<T>>(this->get_connected_edge());
    }
};

template <typename KeyT, typename T>
class MultiSourceProperties : public MultiEdgeHolder<KeyT, T>, public SourcePropertiesBase
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
    std::shared_ptr<IEdgeWritable<T>> get_writable_edge(KeyT edge_key) const
    {
        return std::dynamic_pointer_cast<IEdgeWritable<T>>(this->get_connected_edge(edge_key));
    }
};

template <typename T>
class ReadableProvider : public virtual SourceProperties<T>, public IReadableProvider<T>
{
  public:
    ReadableProvider& operator=(ReadableProvider&& other)
    {
        // Only call concrete class
        SourceProperties<T>::operator=(std::move(other));
    }

  private:
    std::shared_ptr<ReadableEdgeHandle> get_readable_edge_handle() const override
    {
        return ReadableEdgeHandle::from_typeless(SourceProperties<T>::get_edge_connection());
    }
};

template <typename T>
class WritableAcceptor : public virtual SourceProperties<T>, public IWritableAcceptor<T>
{
  public:
    WritableAcceptor& operator=(WritableAcceptor&& other)
    {
        // Only call concrete class
        SourceProperties<T>::operator=(std::move(other));
    }

  private:
    void set_writable_edge_handle(std::shared_ptr<WritableEdgeHandle> ingress) override
    {
        // Do any conversion to the correct type here
        auto adapted_ingress = EdgeBuilder::adapt_writable_edge<T>(ingress);

        SourceProperties<T>::make_edge_connection(adapted_ingress);
    }
};

template <typename T, typename KeyT>
class MultiIngressAcceptor : public virtual MultiSourceProperties<T, KeyT>, public IMultiWritableAcceptor<T, KeyT>
{
  public:
  private:
    void set_writable_edge_handle(KeyT key, std::shared_ptr<WritableEdgeHandle> ingress) override
    {
        // Do any conversion to the correct type here
        auto adapted_ingress = EdgeBuilder::adapt_writable_edge<T>(ingress);

        MultiSourceProperties<T, KeyT>::make_edge_connection(key, adapted_ingress);
    }
};

template <typename T>
class ForwardingEgressProvider : public ReadableProvider<T>
{
  protected:
    class ForwardingEdge : public IEdgeReadable<T>
    {
      public:
        ForwardingEdge(ForwardingEgressProvider<T>& parent) : m_parent(parent) {}

        ~ForwardingEdge() = default;

        channel::Status await_read(T& t) override
        {
            return m_parent.get_next(t);
        }

      private:
        ForwardingEgressProvider<T>& m_parent;
    };

    ForwardingEgressProvider()
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

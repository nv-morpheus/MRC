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
#include "mrc/node/edge.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/type_traits.hpp"
#include "mrc/utils/type_utils.hpp"

#include <memory>
#include <typeindex>

namespace mrc::node {

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

  private:
    // virtual std::shared_ptr<channel::IngressHandle> ingress_handle() = 0;

    friend EdgeBuilder;
};

inline SinkPropertiesBase::~SinkPropertiesBase() = default;

/**
 * @brief Typed SinkProperties provides default implementations dependent only on the type T.
 */
template <typename T>
class SinkProperties : public EdgeHolder<T>, public SinkPropertiesBase
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
    std::shared_ptr<IEdgeReadable<T>> get_readable_edge() const
    {
        return std::dynamic_pointer_cast<IEdgeReadable<T>>(this->get_connected_edge());
    }

  private:
    // inline std::shared_ptr<channel::IngressHandle> ingress_handle() final
    // {
    //     return channel_ingress();
    // }

    // virtual std::shared_ptr<channel::Ingress<T>> channel_ingress() = 0;

    friend EdgeBuilder;
};

template <typename T>
class EgressAcceptor : public virtual SinkProperties<T>, public IEgressAcceptor<T>
{
  public:
    EgressAcceptor& operator=(EgressAcceptor&& other)
    {
        // Only call concrete class
        SinkProperties<T>::operator=(std::move(other));
    }

  private:
    void set_egress_obj(std::shared_ptr<EgressHandleObj> egress) override
    {
        // Do any conversion to the correct type here
        auto adapted_egress = EdgeBuilder::adapt_egress<T>(egress);

        SinkProperties<T>::make_edge_connection(adapted_egress);
    }

    //   private:
    //     using SinkProperties<T>::set_edge;
};

template <typename T>
class IngressProvider : public virtual SinkProperties<T>, public IIngressProvider<T>
{
  public:
    IngressProvider& operator=(IngressProvider&& other)
    {
        // Only call concrete class
        SinkProperties<T>::operator=(std::move(other));
    }

  private:
    std::shared_ptr<IngressHandleObj> get_ingress_obj() const override
    {
        return IngressHandleObj::from_typeless(SinkProperties<T>::get_edge_connection());
    }
};

template <typename T>
class ForwardingIngressProvider : public IngressProvider<T>
{
  protected:
    class ForwardingEdge : public IEdgeWritable<T>
    {
      public:
        ForwardingEdge(ForwardingIngressProvider<T>& parent) : m_parent(parent) {}

        ~ForwardingEdge() = default;

        channel::Status await_write(T&& t) override
        {
            return m_parent.on_next(std::move(t));
        }

      private:
        ForwardingIngressProvider<T>& m_parent;
    };

    ForwardingIngressProvider()
    {
        auto inner_edge = std::make_shared<ForwardingEdge>(*this);

        inner_edge->add_disconnector([this]() {
            // Only call the on_complete if we have been connected
            this->on_complete();
        });

        IngressProvider<T>::init_owned_edge(inner_edge);
    }

    virtual channel::Status on_next(T&& t) = 0;

    virtual void on_complete() {}
};

}  // namespace mrc::node

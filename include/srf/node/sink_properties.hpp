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
#include "srf/node/channel_holder.hpp"
#include "srf/node/edge.hpp"
#include "srf/node/edge_registry.hpp"
#include "srf/node/forward.hpp"
#include "srf/type_traits.hpp"
#include "srf/utils/type_utils.hpp"

#include <memory>
#include <typeindex>

namespace srf::node {

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
        return std::dynamic_pointer_cast<IEdgeReadable<T>>(this->m_edge_connection);
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
    // void set_egress(std::shared_ptr<IEdgeReadable<T>> egress) override
    // {
    //     SinkProperties<T>::set_edge(egress);
    // }

    // void set_egress_typeless(std::shared_ptr<EdgeTag> egress) override
    // {
    //     this->set_egress(std::dynamic_pointer_cast<EdgeReadable<T>>(egress));
    // }

  private:
    void set_egress_obj(std::shared_ptr<EgressHandleObj> egress) override
    {
        SinkProperties<T>::make_edge_connection(egress);
    }

    //   private:
    //     using SinkProperties<T>::set_edge;
};

template <typename T>
class IngressProvider : public virtual SinkProperties<T>, public IIngressProvider<T>
{
  public:
    // std::shared_ptr<IEdgeWritable<T>> get_ingress() const override
    // {
    //     return std::dynamic_pointer_cast<IEdgeWritable<T>>(SinkProperties<T>::get_edge());
    // }

    // std::shared_ptr<EdgeTag> get_ingress_typeless() const override
    // {
    //     return std::dynamic_pointer_cast<EdgeTag>(this->get_ingress());
    // }

  private:
    std::shared_ptr<IngressHandleObj> get_ingress_obj() const override
    {
        return IngressHandleObj::from_typeless(SinkProperties<T>::get_edge_connection());
    }

    //   private:
    //     using SinkProperties<T>::set_edge;
};

template <typename T>
class ForwardingIngressProvider : public IngressProvider<T>
{
  protected:
    class ForwardingEdge : public IEdgeWritable<T>
    {
      public:
        ForwardingEdge(ForwardingIngressProvider<T>& parent) : m_parent(parent) {}

        ~ForwardingEdge()
        {
            m_parent.on_complete();
        }

        channel::Status await_write(T&& t) override
        {
            return m_parent.on_next(std::move(t));
        }

      private:
        ForwardingIngressProvider<T>& m_parent;
    };

    ForwardingIngressProvider()
    {
        IngressProvider<T>::init_edge(std::make_shared<ForwardingEdge>(*this));
    }

    virtual channel::Status on_next(T&& t) = 0;

    virtual void on_complete() {}
};

}  // namespace srf::node

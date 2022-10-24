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
#include "srf/core/utils.hpp"
#include "srf/node/channel_holder.hpp"
#include "srf/node/edge.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/sink_properties.hpp"
#include "srf/type_traits.hpp"

#include <memory>
#include <string>
#include <typeindex>

namespace srf::node {

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
    std::shared_ptr<EdgeWritable<T>> get_writable_edge() const
    {
        return std::dynamic_pointer_cast<EdgeWritable<T>>(this->m_set_edge);
    }
};

template <typename T, typename KeyT>
class MultiSourceProperties : public MultiEdgeHolder<T, KeyT>, public SourcePropertiesBase
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
    std::shared_ptr<EdgeWritable<T>> get_writable_edge(KeyT edge_key) const
    {
        return std::dynamic_pointer_cast<EdgeWritable<T>>(this->get_edge_pair(edge_key).second);
    }
};

template <typename T>
class EgressProvider : public virtual SourceProperties<T>, public IEgressProvider<T>
{
  public:
    std::shared_ptr<EdgeReadable<T>> get_egress() const override
    {
        return std::dynamic_pointer_cast<EdgeReadable<T>>(SourceProperties<T>::get_edge());
    }

    std::shared_ptr<EdgeTag> get_egress_typeless() const override
    {
        return this->get_egress();
    }

  private:
    using SourceProperties<T>::set_edge;
};

template <typename T>
class IngressAcceptor : public virtual SourceProperties<T>, public IIngressAcceptor<T>
{
  public:
    void set_ingress(std::shared_ptr<EdgeWritable<T>> ingress) override
    {
        SourceProperties<T>::set_edge(ingress);
    }

    void set_ingress_typeless(std::shared_ptr<EdgeTag> ingress) override
    {
        this->set_ingress(std::dynamic_pointer_cast<EdgeWritable<T>>(ingress));
    }

  private:
    using SourceProperties<T>::set_edge;
};

}  // namespace srf::node

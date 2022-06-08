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

#include <srf/channel/ingress.hpp>
#include <srf/core/utils.hpp>
#include <srf/node/edge.hpp>
#include <srf/node/edge_registry.hpp>
#include <srf/node/forward.hpp>
#include <srf/type_traits.hpp>

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
    virtual std::shared_ptr<channel::IngressHandle> ingress_handle() = 0;

    friend SinkTypeErased;
};

inline SinkPropertiesBase::~SinkPropertiesBase() = default;

/**
 * @brief Typed SinkProperties provides default implementations dependent only on the type T.
 */
template <typename T>
class SinkProperties : public virtual SinkPropertiesBase
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

  private:
    inline std::shared_ptr<channel::IngressHandle> ingress_handle() final
    {
        return channel_ingress();
    }

    virtual std::shared_ptr<channel::Ingress<T>> channel_ingress() = 0;

    friend EdgeBuilder;
};

class SinkTypeErased : public virtual SinkPropertiesBase
{
  protected:
    using SinkPropertiesBase::ingress_handle;

    virtual std::shared_ptr<channel::IngressHandle> ingress_for_source_type(std::type_index source_type)
    {
        // Get the converter function
        auto converter_fn = EdgeRegistry::find_converter(source_type, sink_type());

        // Build the edge from our channel
        return converter_fn(this->ingress_handle());
    }

    friend SourceTypeErased;
};

}  // namespace srf::node

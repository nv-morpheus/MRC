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

#include "srf/node/port_registry.hpp"
#include "srf/segment/egress_port.hpp"
#include "srf/segment/ingress_port.hpp"
#include "srf/segment/object.hpp"

#include <typeinfo>

namespace srf::pysrf {

template <typename T, typename U>
struct WrappedType
{
    using wrapped_type_t = T;
};

template <typename T>
struct WrappedType<T, std::true_type>
{
    using wrapped_type_t = typename T::element_type;
};

struct PortBuilderUtil
{
    template <typename IngressDataT>
    static node::PortUtil::ingress_tuple_t create_ingress_builders()
    {
        return std::tuple(
            [](SegmentAddress address, PortName name) {
                VLOG(2) << "Building raw ingress port: " << type_name<IngressDataT>();
                auto ingress_port = std::make_shared<segment::IngressPort<IngressDataT>>(address, name);

                return ingress_port;
            },
            [](SegmentAddress address, PortName name) {
                VLOG(2) << "Building sp wrapped ingress port: " << type_name<IngressDataT>();
                auto ingress_port =
                    std::make_shared<segment::IngressPort<std::shared_ptr<IngressDataT>>>(address, name);

                return ingress_port;
            });
    }

    template <typename EgressDataT>
    static node::PortUtil::egress_tuple_t create_egress_builders()
    {
        return std::tuple(
            [](SegmentAddress address, PortName name) {
                VLOG(2) << "Building raw egress port: " << type_name<EgressDataT>();
                auto egress_port = std::make_shared<segment::EgressPort<EgressDataT>>(address, name);

                return egress_port;
            },
            [](SegmentAddress address, PortName name) {
                VLOG(2) << "Building sp wrapped egress port: " << type_name<EgressDataT>();
                auto egress_port = std::make_shared<segment::EgressPort<std::shared_ptr<EgressDataT>>>(address, name);

                return egress_port;
            });
    }

    template <typename IngressDataT>
    static node::PortUtil::ingress_caster_tuple_t create_ingress_casters()
    {
        return std::tuple(
            [](std::shared_ptr<srf::segment::IngressPortBase> base) -> std::shared_ptr<segment::ObjectProperties> {
                VLOG(2) << "Attempting dynamic Ingress cast for: " << type_name<decltype(base)>() << " into "
                        << type_name<segment::Object<node::SourceProperties<IngressDataT>>>();

                return std::dynamic_pointer_cast<segment::Object<node::SourceProperties<IngressDataT>>>(base);
            },
            [](std::shared_ptr<srf::segment::IngressPortBase> base) -> std::shared_ptr<segment::ObjectProperties> {
                VLOG(2) << "Attempting dynamic Ingress cast for: " << type_name<decltype(base)>() << " into "
                        << type_name<segment::Object<node::SourceProperties<std::shared_ptr<IngressDataT>>>>();

                return std::dynamic_pointer_cast<
                    segment::Object<node::SourceProperties<std::shared_ptr<IngressDataT>>>>(base);
            });
    }

    template <typename EgressDataT>
    static node::PortUtil::egress_caster_tuple_t create_egress_casters()
    {
        return std::tuple(
            [](std::shared_ptr<srf::segment::EgressPortBase> base) -> std::shared_ptr<segment::ObjectProperties> {
                VLOG(2) << "Attempting dynamic Egress cast for: " << type_name<decltype(base)>() << " into "
                        << type_name<segment::Object<node::SinkProperties<EgressDataT>>>();

                return std::dynamic_pointer_cast<segment::Object<node::SinkProperties<EgressDataT>>>(base);
            },
            [](std::shared_ptr<srf::segment::EgressPortBase> base) -> std::shared_ptr<segment::ObjectProperties> {
                VLOG(2) << "Attempting dynamic Egress cast for: " << type_name<decltype(base)>() << " into "
                        << type_name<segment::Object<node::SinkProperties<std::shared_ptr<EgressDataT>>>>();

                return std::dynamic_pointer_cast<segment::Object<node::SinkProperties<std::shared_ptr<EgressDataT>>>>(
                    base);
            });
    }

    template <typename PortDataTypeT>
    static void register_port_util()
    {
        using port_type_t = PortDataTypeT;
        using port_dtype_t =
            typename WrappedType<PortDataTypeT, typename is_smart_ptr<PortDataTypeT>::type>::wrapped_type_t;

        std::type_index type_idx = typeid(port_dtype_t);

        if (!srf::node::PortRegistry::has_port_util(type_idx))
        {
            VLOG(2) << "Registering PySRF port util for: " << type_name<port_type_t>() << " "
                    << "=> " << type_name<port_dtype_t>() << " " << type_idx.hash_code();

            auto port_util = std::make_shared<srf::node::PortUtil>(typeid(PortDataTypeT));

            port_util->m_ingress_builders = create_ingress_builders<port_dtype_t>();
            port_util->m_egress_builders  = create_egress_builders<port_dtype_t>();
            port_util->m_ingress_casters  = create_ingress_casters<port_dtype_t>();
            port_util->m_egress_casters   = create_egress_casters<port_dtype_t>();

            node::PortRegistry::register_port_util(port_util);
        }
    }
};

template <typename IngressT>
struct AutoRegIngressPort
{
    AutoRegIngressPort()
    {
        static bool _init = register_port();
    }

    static bool register_port()
    {
        PortBuilderUtil::register_port_util<IngressT>();

        return true;
    }
};

template <typename EgressT>
struct AutoRegEgressPort
{
    AutoRegEgressPort()
    {
        static bool _init = register_port();
    }

    static bool register_port()
    {
        PortBuilderUtil::register_port_util<EgressT>();

        return true;
    }
};

}  // namespace srf::pysrf

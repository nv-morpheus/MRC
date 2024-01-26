/*
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

#include "mrc/codable/fundamental_types.hpp"
#include "mrc/edge/edge_connector.hpp"
#include "mrc/manifold/factory.hpp"
#include "mrc/node/port_registry.hpp"
#include "mrc/runtime/remote_descriptor.hpp"
#include "mrc/segment/egress_port.hpp"
#include "mrc/segment/ingress_port.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/utils/type_utils.hpp"

#include <memory>
#include <type_traits>
#include <typeinfo>

namespace mrc::node {

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
    static PortUtil::ingress_tuple_t create_ingress_builders()
    {
        // Check if we are default constructible. If not, we cannot register the port since channels need to create
        // objects
        if constexpr (std::is_default_constructible_v<IngressDataT>)
        {
            return PortUtil::ingress_tuple_t(
                [](SegmentAddress address, PortName name) {
                    VLOG(2) << "Building raw ingress port: " << type_name<IngressDataT>();
                    auto ingress_port = std::make_shared<segment::IngressPort<IngressDataT>>(address, name);

                    return ingress_port;
                },
                [](SegmentAddress address, PortName name) {
                    VLOG(2) << "Building sp wrapped ingress port: " << type_name<IngressDataT>();
                    auto ingress_port = std::make_shared<segment::IngressPort<std::shared_ptr<IngressDataT>>>(address,
                                                                                                              name);

                    return ingress_port;
                });
        }
        else
        {
            return PortUtil::ingress_tuple_t(nullptr, [](SegmentAddress address, PortName name) {
                VLOG(2) << "Building sp wrapped ingress port: " << type_name<IngressDataT>();
                auto ingress_port = std::make_shared<segment::IngressPort<std::shared_ptr<IngressDataT>>>(address,
                                                                                                          name);

                return ingress_port;
            });
        }
    }

    template <typename EgressDataT>
    static PortUtil::egress_tuple_t create_egress_builders()
    {
        // Check if we are default constructible. If not, we cannot register the port since channels need to create
        // objects
        if constexpr (std::is_default_constructible_v<EgressDataT>)
        {
            return PortUtil::egress_tuple_t(
                [](SegmentAddress address, PortName name) {
                    VLOG(2) << "Building raw egress port: " << type_name<EgressDataT>();
                    auto egress_port = std::make_shared<segment::EgressPort<EgressDataT>>(address, name);

                    return egress_port;
                },
                [](SegmentAddress address, PortName name) {
                    VLOG(2) << "Building sp wrapped egress port: " << type_name<EgressDataT>();
                    auto egress_port = std::make_shared<segment::EgressPort<std::shared_ptr<EgressDataT>>>(address,
                                                                                                           name);

                    return egress_port;
                });
        }
        else
        {
            return PortUtil::egress_tuple_t(nullptr, [](SegmentAddress address, PortName name) {
                VLOG(2) << "Building sp wrapped egress port: " << type_name<EgressDataT>();
                auto egress_port = std::make_shared<segment::EgressPort<std::shared_ptr<EgressDataT>>>(address, name);

                return egress_port;
            });
        }
    }

    // template <typename IngressDataT>
    // static PortUtil::ingress_caster_tuple_t create_ingress_casters()
    // {
    //     return std::tuple(
    //         [](std::shared_ptr<mrc::segment::IngressPortBase> base) -> std::shared_ptr<segment::ObjectProperties> {
    //             VLOG(2) << "Attempting dynamic Ingress cast for: " << type_name<decltype(base)>() << " into "
    //                     << type_name<segment::Object<node::RxSourceBase<IngressDataT>>>();

    //             return std::dynamic_pointer_cast<segment::Object<node::RxSourceBase<IngressDataT>>>(base);
    //         },
    //         [](std::shared_ptr<mrc::segment::IngressPortBase> base) -> std::shared_ptr<segment::ObjectProperties> {
    //             VLOG(2) << "Attempting dynamic Ingress cast for: " << type_name<decltype(base)>() << " into "
    //                     << type_name<segment::Object<node::RxSourceBase<std::shared_ptr<IngressDataT>>>>();

    //             return std::dynamic_pointer_cast<segment::Object<node::RxSourceBase<std::shared_ptr<IngressDataT>>>>(
    //                 base);
    //         });
    // }

    // template <typename EgressDataT>
    // static PortUtil::egress_caster_tuple_t create_egress_casters()
    // {
    //     return std::tuple(
    //         [](std::shared_ptr<mrc::segment::EgressPortBase> base) -> std::shared_ptr<segment::ObjectProperties> {
    //             VLOG(2) << "Attempting dynamic Egress cast for: " << type_name<decltype(base)>() << " into "
    //                     << type_name<segment::Object<node::RxSinkBase<EgressDataT>>>();

    //             return std::dynamic_pointer_cast<segment::Object<node::RxSinkBase<EgressDataT>>>(base);
    //         },
    //         [](std::shared_ptr<mrc::segment::EgressPortBase> base) -> std::shared_ptr<segment::ObjectProperties> {
    //             VLOG(2) << "Attempting dynamic Egress cast for: " << type_name<decltype(base)>() << " into "
    //                     << type_name<segment::Object<node::RxSinkBase<std::shared_ptr<EgressDataT>>>>();

    //             return
    //             std::dynamic_pointer_cast<segment::Object<node::RxSinkBase<std::shared_ptr<EgressDataT>>>>(base);
    //         });
    // }

    template <typename T>
    static segment::manifold_initializer_fn_t create_manifold_builder()
    {
        return [](std::string name, runnable::IRunnableResources& resources) {
            return manifold::Factory<T>::make_manifold(std::move(name), resources);
        };
    }

    template <typename PortDataTypeT>
    static void register_port_util()
    {
        using port_type_t = PortDataTypeT;
        using port_dtype_t =
            typename WrappedType<PortDataTypeT, typename is_smart_ptr<PortDataTypeT>::type>::wrapped_type_t;

        std::type_index type_idx = typeid(port_dtype_t);

        if (!mrc::node::PortRegistry::has_port_util(type_idx))
        {
            // VLOG(2) << "Registering PyMRC port util for: " << type_name<port_type_t>() << " "
            //         << "=> " << type_name<port_dtype_t>() << " " << type_idx.hash_code();

            auto port_util = std::make_shared<PortUtil>(typeid(port_dtype_t));

            port_util->ingress_builders = create_ingress_builders<port_dtype_t>();
            port_util->egress_builders  = create_egress_builders<port_dtype_t>();
            // port_util->ingress_casters  = create_ingress_casters<port_dtype_t>();
            // port_util->egress_casters   = create_egress_casters<port_dtype_t>();
            port_util->manifold_builder_fn = create_manifold_builder<port_dtype_t>();

            node::PortRegistry::register_port_util(port_util);

            // Register the necessary edge converters to convert to/from Descriptors

            // T -> ValueDescriptor
            edge::EdgeConnector<PortDataTypeT, std::unique_ptr<runtime::ValueDescriptor>>::register_converter(
                [](PortDataTypeT&& data) {
                    // Convert from T -> TypedValueDescriptor<T> -> ValueDescriptor
                    return runtime::TypedValueDescriptor<PortDataTypeT>::create(std::move(data));
                });

            // // T -> ResidentDescriptor<T>
            // edge::EdgeConnector<PortDataTypeT, std::unique_ptr<runtime::ResidentDescriptor<PortDataTypeT>>>::
            //     register_converter([](PortDataTypeT&& data) {
            //         return std::make_unique<runtime::ResidentDescriptor<PortDataTypeT>>(std::move(data));
            //     });

            // // T -> CodedDescriptor<T>
            // edge::EdgeConnector<PortDataTypeT, std::unique_ptr<runtime::CodedDescriptor<PortDataTypeT>>>::
            //     register_converter([](PortDataTypeT&& data) {
            //         return std::make_unique<runtime::CodedDescriptor<PortDataTypeT>>(std::move(data));
            //     });

            // Descriptor -> T
            edge::EdgeConnector<std::unique_ptr<runtime::ValueDescriptor>, PortDataTypeT>::register_converter(
                [](std::unique_ptr<runtime::ValueDescriptor>&& descriptor) {
                    // Move into a temp object so it goes out of scope
                    auto temp_descriptor = std::move(descriptor);

                    return std::move(*temp_descriptor).release_value<PortDataTypeT>();
                });

            // // ResidentDescriptor<T> -> Descriptor
            // edge::EdgeConnector<std::unique_ptr<runtime::ResidentDescriptor<PortDataTypeT>>,
            //                     std::unique_ptr<runtime::Descriptor>>::register_converter();

            // // CodedDescriptor<T> -> Descriptor
            // edge::EdgeConnector<std::unique_ptr<runtime::CodedDescriptor<PortDataTypeT>>,
            //                     std::unique_ptr<runtime::Descriptor>>::register_converter();
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

}  // namespace mrc::node

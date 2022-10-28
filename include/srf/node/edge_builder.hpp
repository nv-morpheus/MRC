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
#include "srf/node/edge_properties.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/sink_properties.hpp"
#include "srf/node/source_properties.hpp"

#include <glog/logging.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <typeindex>

namespace srf::node {

struct EdgeBuilder final
{
    /**
     * @brief Attempt to look-up a registered ingress adapter given the source and sink properties. If one exists
     * use it, otherwise fall back to the default adapter lookup.
     * @param source
     * @param sink
     * @param ingress_handle
     * @return Ingress handle constructed by the adapter
     */
    static std::shared_ptr<IEdgeWritableBase> ingress_adapter_for_sink(
        IIngressAcceptorBase& source, IIngressProviderBase& sink, std::shared_ptr<IEdgeWritableBase> ingress_handle);

    static std::shared_ptr<IngressHandleObj> adapt_ingress(const EdgeTypePair& target_type,
                                                           std::shared_ptr<IngressHandleObj> ingress);

    /**
     * @brief Attempt to look-up a registered ingress adapter for the given source type and sink properties. If one
     * exists, use it, otherwise fall back to default.
     * @param source
     * @param sink
     * @param ingress_handle
     * @return
     */
    static std::shared_ptr<IEdgeWritableBase> ingress_for_source_type(
        std::type_index source_type, IIngressProviderBase& sink, std::shared_ptr<IEdgeWritableBase> ingress_handle);

    /**
     *
     * @param source
     * @param sink
     * @param allow_narrowing
     */
    static void make_edge_typeless(SourcePropertiesBase& source, SinkPropertiesBase& sink, bool allow_narrowing = true);

    static void make_edge_ingress_typeless(IIngressAcceptorBase& source,
                                           IIngressProviderBase& sink,
                                           bool allow_narrowing = true);

    static void make_edge_egress_typeless(IEgressProviderBase& source,
                                          IEgressAcceptorBase& sink,
                                          bool allow_narrowing = true);

    /**
     *
     * @tparam SourceT
     * @tparam SinkT
     * @tparam AllowNarrowingV
     * @param source
     * @param sink
     */
    template <typename SourceT, typename SinkT = SourceT, bool AllowNarrowingV = true>
    static void make_edge(SourceProperties<SourceT>& source, SinkProperties<SinkT>& sink)
    {
        constexpr bool IsConvertable = std::is_convertible_v<SourceT, SinkT>;
        constexpr bool LessBits      = sizeof(SourceT) > sizeof(SinkT);  // Sink requires more bits than source.
        constexpr bool FloatToInt    = std::is_floating_point_v<SourceT> && std::is_integral_v<SinkT>;  // float -> int
        constexpr bool SignedToUnsigned = std::is_signed_v<SourceT> && !std::is_signed_v<SinkT>;  // signed -> unsigned
        constexpr bool UnsignedToSignedLessBits =
            !std::is_signed_v<SourceT> && std::is_signed_v<SinkT> &&
            (sizeof(SourceT) == sizeof(SinkT));  // Unsigned component could exceed signed limits

        // If its convertable but may result in loss of data, it requires narrowing
        constexpr bool RequiresNarrowing =
            IsConvertable && (LessBits || FloatToInt || SignedToUnsigned || UnsignedToSignedLessBits);

        std::shared_ptr<channel::IngressHandle> edge;

        if constexpr (std::is_same_v<SourceT, SinkT>)
        {
            // Easy case, both nodes are the same type, no conversion required.
            // edge = sink.channel_ingress();
        }
        else if constexpr (IsConvertable && !RequiresNarrowing)
        {
            // Static lookup with implicit conversion. No narrowing required
            edge = std::make_shared<node::Edge<SourceT, SinkT>>(sink.channel_ingress());
        }
        else if constexpr (RequiresNarrowing && AllowNarrowingV)
        {
            // Static lookup with implicit conversion. Narrowing required
            LOG(WARNING) << "WARNING: Automatic edge conversion will result in a narrowing cast.";
            edge = std::make_shared<node::Edge<SourceT, SinkT>>(sink.channel_ingress());
        }
        else
        {
            LOG(FATAL) << "No dynamic lookup available for statically typed objects";
        }

        // source.complete_edge(edge);
    }

    template <typename T>
    static void make_edge(ChannelProvider<T>& source, ChannelAcceptor<T>& sink)
    {
        sink.set_channel(source.channel());
    }

    template <typename SourceT, typename SinkT = SourceT, bool AllowNarrowingV = true>
    static void make_edge_ingress(IIngressAcceptor<SourceT>& source, IIngressProvider<SinkT>& sink)
    {
        constexpr bool IsConvertable = std::is_convertible_v<SourceT, SinkT>;
        constexpr bool LessBits      = sizeof(SourceT) > sizeof(SinkT);  // Sink requires more bits than source.
        constexpr bool FloatToInt    = std::is_floating_point_v<SourceT> && std::is_integral_v<SinkT>;  // float -> int
        constexpr bool SignedToUnsigned = std::is_signed_v<SourceT> && !std::is_signed_v<SinkT>;  // signed -> unsigned
        constexpr bool UnsignedToSignedLessBits =
            !std::is_signed_v<SourceT> && std::is_signed_v<SinkT> &&
            (sizeof(SourceT) == sizeof(SinkT));  // Unsigned component could exceed signed limits

        // If its convertable but may result in loss of data, it requires narrowing
        constexpr bool RequiresNarrowing =
            IsConvertable && (LessBits || FloatToInt || SignedToUnsigned || UnsignedToSignedLessBits);

        std::shared_ptr<IngressHandleObj> edge;

        if constexpr (std::is_same_v<SourceT, SinkT>)
        {
            // Easy case, both nodes are the same type, no conversion required.
            edge = sink.get_ingress_obj();
        }
        else if constexpr (IsConvertable)
        {
            if constexpr (RequiresNarrowing && AllowNarrowingV)
            {
                // Static lookup with implicit conversion. Narrowing required
                LOG(WARNING) << "WARNING: Automatic edge conversion will result in a narrowing cast.";
            }

            // Unpack the ingress object
            auto sink_typed = sink.get_ingress_obj()->template get_ingress_typed<SinkT>();

            // Make a converting edge
            auto converting_edge = std::make_shared<ConvertingEdgeWritable<SourceT, SinkT>>(sink_typed);

            // Repack the object back into the handle
            edge = std::make_shared<IngressHandleObj>(converting_edge);
        }
        else
        {
            LOG(FATAL) << "No dynamic lookup available for statically typed objects";
        }

        source.set_ingress_obj(edge);
    }

    template <typename SourceT, typename SinkT = SourceT, bool AllowNarrowingV = true>
    static void make_edge_egress(IEgressProvider<SourceT>& source, IEgressAcceptor<SinkT>& sink)
    {
        constexpr bool IsConvertable = std::is_convertible_v<SinkT, SourceT>;
        constexpr bool LessBits      = sizeof(SinkT) > sizeof(SourceT);  // Sink requires more bits than source.
        constexpr bool FloatToInt    = std::is_floating_point_v<SourceT> && std::is_integral_v<SinkT>;  // float -> int
        constexpr bool SignedToUnsigned = std::is_signed_v<SinkT> && !std::is_signed_v<SourceT>;  // signed -> unsigned
        constexpr bool UnsignedToSignedLessBits =
            !std::is_signed_v<SinkT> && std::is_signed_v<SourceT> &&
            (sizeof(SourceT) == sizeof(SinkT));  // Unsigned component could exceed signed limits

        // If its convertable but may result in loss of data, it requires narrowing
        constexpr bool RequiresNarrowing =
            IsConvertable && (LessBits || FloatToInt || SignedToUnsigned || UnsignedToSignedLessBits);

        std::shared_ptr<EgressHandleObj> edge;

        if constexpr (std::is_same_v<SourceT, SinkT>)
        {
            // Easy case, both nodes are the same type, no conversion required.
            edge = source.get_egress_obj();
        }
        else if constexpr (IsConvertable)
        {
            if constexpr (RequiresNarrowing && AllowNarrowingV)
            {
                // Static lookup with implicit conversion. Narrowing required
                LOG(WARNING) << "WARNING: Automatic edge conversion will result in a narrowing cast.";
            }

            // Unpack the ingress object
            auto source_typed = source.get_egress_obj()->template get_egress_typed<SourceT>();

            // Make a converting edge
            auto converting_edge = std::make_shared<ConvertingEdgeReadable<SourceT, SinkT>>(source_typed);

            // Repack the object back into the handle
            edge = std::make_shared<EgressHandleObj>(converting_edge);
        }
        else
        {
            LOG(FATAL) << "No dynamic lookup available for statically typed objects";
        }

        sink.set_egress_obj(edge);
    }
};

template <typename SourceT, typename SinkT = SourceT>
void make_edge(SourceProperties<SourceT>& source, SinkProperties<SinkT>& sink)
{
    EdgeBuilder::make_edge(source, sink);
}

template <typename SourceT, typename SinkT = SourceT>
void make_edge(ChannelProvider<SourceT>& source, ChannelAcceptor<SinkT>& sink)
{
    EdgeBuilder::make_edge(source, sink);
}

template <typename SourceT, typename SinkT>
void make_edge(SourceT& source, SinkT& sink)
{
    using source_full_t = SourceT;
    using sink_full_t   = SinkT;

    if constexpr (is_base_of_template<IIngressAcceptor, source_full_t>::value &&
                  is_base_of_template<IIngressProvider, sink_full_t>::value)
    {
        // Call the typed version for ingress provider/acceptor
        EdgeBuilder::make_edge_ingress(source, sink);
    }
    else if constexpr (is_base_of_template<IEgressProvider, source_full_t>::value &&
                       is_base_of_template<IEgressAcceptor, sink_full_t>::value)
    {
        // Call the typed version for egress provider/acceptor
        EdgeBuilder::make_edge_egress(source, sink);
    }
    else
    {
        static_assert(!sizeof(source_full_t),
                      "Arguments to make_edge were incorrect. Ensure you are providing either "
                      "IngressAcceptor->IngressProvider or EgressProvider->EgressAcceptor");
    }
}

template <typename SourceT, typename SinkT>
void make_edge_typeless(SourceT& source, SinkT& sink)
{
    using source_full_t = SourceT;
    using sink_full_t   = SinkT;

    if constexpr (std::is_base_of_v<IIngressAcceptorBase, source_full_t> &&
                  std::is_base_of_v<IIngressProviderBase, sink_full_t>)
    {
        EdgeBuilder::make_edge_ingress_typeless(source, sink);
    }
    else if constexpr (std::is_base_of_v<IEgressProviderBase, source_full_t> &&
                       std::is_base_of_v<IEgressAcceptorBase, sink_full_t>)
    {
        EdgeBuilder::make_edge_egress_typeless(source, sink);
    }
    else
    {
        static_assert(!sizeof(source_full_t),
                      "Arguments to make_edge were incorrect. Ensure you are providing either "
                      "IngressAcceptor->IngressProvider or EgressProvider->EgressAcceptor");
    }
}

template <typename SourceT, typename SinkT>
void operator|(SourceProperties<SourceT>& source, SinkProperties<SinkT>& sink)
{
    EdgeBuilder::make_edge(source, sink);
}

}  // namespace srf::node

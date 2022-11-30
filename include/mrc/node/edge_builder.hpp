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

#include "mrc/channel/ingress.hpp"
#include "mrc/node/edge_properties.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"

#include <glog/logging.h>

#include <memory>
#include <sstream>
#include <typeindex>

namespace mrc::node {

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
    static std::shared_ptr<channel::IngressHandle> ingress_adapter_for_sink(
        mrc::node::SourcePropertiesBase& source,
        mrc::node::SinkPropertiesBase& sink,
        std::shared_ptr<channel::IngressHandle> ingress_handle);

    /**
     * @brief Attempt to look-up a registered ingress adapter for the given source type and sink properties. If one
     * exists, use it, otherwise fall back to default.
     * @param source
     * @param sink
     * @param ingress_handle
     * @return
     */
    static std::shared_ptr<channel::IngressHandle> ingress_for_source_type(
        std::type_index source_type,
        mrc::node::SinkPropertiesBase& sink,
        std::shared_ptr<channel::IngressHandle> ingress_handle);

    /**
     *
     * @param source
     * @param sink
     * @param allow_narrowing
     */
    static void make_edge_typeless(SourcePropertiesBase& source, SinkPropertiesBase& sink, bool allow_narrowing = true);

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
            edge = sink.channel_ingress();
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
            // todo(cpp20) - use templated lambda to trigger the static fail - make a MRC_STATIC_FAIL macro
            static_assert(!sizeof(SourceT), "No dynamic lookup available for statically typed objects");
        }

        source.complete_edge(edge);
    }

    template <typename T>
    static void make_edge(ChannelProvider<T>& source, ChannelAcceptor<T>& sink)
    {
        sink.set_channel(source.channel());
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
void operator|(SourceProperties<SourceT>& source, SinkProperties<SinkT>& sink)
{
    EdgeBuilder::make_edge(source, sink);
}

}  // namespace mrc::node

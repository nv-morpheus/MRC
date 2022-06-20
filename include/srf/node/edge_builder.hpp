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
#include <srf/node/edge.hpp>
#include <srf/node/edge_adaptor.hpp>
#include <srf/node/sink_properties.hpp>
#include <srf/node/source_properties.hpp>
#include <srf/utils/type_utils.hpp>

#include <memory>
#include <mutex>
#include <sstream>
#include <type_traits>
#include <typeindex>
#include <typeinfo>

namespace srf::node {

struct EdgeBuilder final
{
    static std::mutex s_edge_adaptor_mutex;
    static std::map<std::string, std::shared_ptr<EdgeAdaptorBase>> s_edge_adaptors;

    template <typename EdgeAdaptorT>
    static void register_edge_adaptor(const std::string& constructor_id)
    {
        static_assert(std::is_base_of_v<EdgeAdaptorBase, EdgeAdaptorT>);

        std::lock_guard<std::mutex> lock(s_edge_adaptor_mutex);
        if (s_edge_adaptors.find(constructor_id) != s_edge_adaptors.end())
        {
            std::stringstream sstream;
            sstream << "Attempted to register existing edge adaptor: " + constructor_id;

            LOG(WARNING) << sstream.str();
            throw std::runtime_error(sstream.str());
        }

        s_edge_adaptors[constructor_id] = std::make_shared<EdgeAdaptorT>();
    }

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
            LOG(FATAL) << "No dynamic lookup available for statically typed objects";
        }

        source.complete_edge(edge);
    }

    static void make_edge_typeless(SourcePropertiesBase& source,
                                   SinkPropertiesBase& sink,
                                   const std::string& constructor_id = "default",
                                   bool allow_narrowing              = true);
};

template <typename SourceT, typename SinkT = SourceT>
void make_edge(SourceProperties<SourceT>& source, SinkProperties<SinkT>& sink)
{
    EdgeBuilder::make_edge(source, sink);
}

template <typename SourceT, typename SinkT>
void operator|(SourceProperties<SourceT>& source, SinkProperties<SinkT>& sink)
{
    EdgeBuilder::make_edge(source, sink);
}

}  // namespace srf::node

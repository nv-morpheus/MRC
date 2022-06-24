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

#include <srf/node/edge_adaptor_registry.hpp>
#include <srf/node/edge_builder.hpp>
#include <srf/node/edge_registry.hpp>

namespace srf::node {
void EdgeBuilder::make_edge_typeless(SourcePropertiesBase& source, SinkPropertiesBase& sink, bool allow_narrowing)
{
    source.complete_edge(EdgeBuilder::ingress_adaptor_for_sink(source, sink, sink.ingress_handle()));
}

std::shared_ptr<channel::IngressHandle> EdgeBuilder::ingress_adaptor_for_sink(
    srf::node::SourcePropertiesBase& source,
    srf::node::SinkPropertiesBase& sink,
    std::shared_ptr<channel::IngressHandle> ingress_handle)
{
    VLOG(2) << "Looking for edge adaptor: (" << source.source_type_name() << ", " << sink.sink_type_name() << ")";
    VLOG(2) << "- (" << source.source_type_hash() << ", " << sink.sink_type_hash() << ")";
    if (EdgeAdaptorRegistry::has_source_adaptor(source.source_type()))
    {
        auto adaptor = EdgeAdaptorRegistry::find_source_adaptor(source.source_type());
        return adaptor(source, sink, sink.ingress_handle());
    }

    return default_ingress_adaptor_for_sink(source, sink, ingress_handle);
}

std::shared_ptr<channel::IngressHandle> EdgeBuilder::default_ingress_adaptor_for_sink(
    srf::node::SourcePropertiesBase& source,
    srf::node::SinkPropertiesBase& sink,
    std::shared_ptr<channel::IngressHandle> ingress_handle)

{
    auto fn_converter = srf::node::EdgeRegistry::find_converter(source.source_type(), sink.sink_type());
    return fn_converter(ingress_handle);
}

std::shared_ptr<channel::IngressHandle> EdgeBuilder::default_ingress_for_source_type(
    std::type_index source_type,
    srf::node::SinkPropertiesBase& sink,
    std::shared_ptr<channel::IngressHandle> ingress_handle)
{
    auto fn_converter = srf::node::EdgeRegistry::find_converter(source_type, sink.sink_type());
    return fn_converter(ingress_handle);
}

std::shared_ptr<channel::IngressHandle> EdgeBuilder::ingress_for_source_type(
    std::type_index source_type,
    srf::node::SinkPropertiesBase& sink,
    std::shared_ptr<channel::IngressHandle> ingress_handle)
{
    if (EdgeAdaptorRegistry::has_sink_adaptor(sink.sink_type()))
    {
        auto adaptor = EdgeAdaptorRegistry::find_sink_adaptor(sink.sink_type());
        return adaptor(source_type, sink, sink.ingress_handle());
    }

    return default_ingress_for_source_type(source_type, sink, ingress_handle);
}

}  // namespace srf::node
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

#include "srf/node/edge_builder.hpp"

#include "srf/channel/ingress.hpp"
#include "srf/exceptions//runtime_error.hpp"
#include "srf/node/channel_holder.hpp"
#include "srf/node/edge_adapter_registry.hpp"
#include "srf/node/edge_registry.hpp"
#include "srf/node/sink_properties.hpp"
#include "srf/node/source_properties.hpp"
#include "srf/utils/type_utils.hpp"

#include <glog/logging.h>

#include <functional>
#include <memory>
#include <string>
#include <typeindex>

namespace srf::node {
void EdgeBuilder::make_edge_typeless(SourcePropertiesBase& source, SinkPropertiesBase& sink, bool allow_narrowing)
{
    // source.complete_edge(EdgeBuilder::ingress_adapter_for_sink(source, sink, sink.ingress_handle()));
}

void EdgeBuilder::make_edge_ingress_typeless(IIngressAcceptorBase& source,
                                             IIngressProviderBase& sink,
                                             bool allow_narrowing)
{
    // Get the ingress
    auto ingress = sink.get_ingress_obj();

    // Set to the source
    source.set_ingress_obj(ingress);
}

void EdgeBuilder::make_edge_egress_typeless(IEgressProviderBase& source,
                                            IEgressAcceptorBase& sink,
                                            bool allow_narrowing)
{
    EdgeTypePair source_type_pair = source.egress_provider_type();
    EdgeTypePair sink_type_pair   = sink.egress_acceptor_type();

    // Get the ingress
    auto egress = source.get_egress_obj();

    // Now try and loop over any ingress adaptors for the sink
    // auto adapted_egress = EdgeBuilder::adapt_egress(sink_type_pair, egress);

    // Try it again in case we need a sink adaptor then a source adaptor (Short circuits if we are already there)
    // adapted_egress = EdgeBuilder::adapt_egress(source_type_pair, adapted_egress);

    // Convert if neccessary
    // auto ingress_adapted = EdgeBuilder::ingress_adapter_for_sink(source, sink, ingress);

    // Set to the source
    // sink.set_egress_obj(adapted_egress);
}

// std::shared_ptr<IEdgeWritableBase> EdgeBuilder::ingress_adapter_for_sink(
//     IIngressAcceptorBase& source, IIngressProviderBase& sink, std::shared_ptr<IEdgeWritableBase> ingress_handle)
// {
//     VLOG(2) << "Looking for edge adapter: (" << type_name(source.ingress_acceptor_type()) << ", "
//             << type_name(sink.ingress_provider_type()) << ")";
//     VLOG(2) << "- (" << source.ingress_acceptor_type().hash_code() << ", " <<
//     sink.ingress_provider_type().hash_code()
//             << ")";

//     if (EdgeAdapterRegistry::has_source_adapter(source.ingress_acceptor_type()))
//     {
//         auto adapter = EdgeAdapterRegistry::find_source_adapter(source.ingress_acceptor_type());

//         // Try and build the handle
//         auto handle = adapter(source, sink, sink.get_ingress_typeless());
//         if (handle)
//         {
//             return handle;
//         }
//     }

//     // Fallback -- probably fail
//     auto fn_converter = srf::node::EdgeRegistry::find_converter(source.source_type(), sink.sink_type());
//     return fn_converter(ingress_handle);
// }

// std::shared_ptr<IngressHandleObj> EdgeBuilder::adapt_ingress(const EdgeTypePair& target_type,
//                                                              std::shared_ptr<IngressHandleObj> ingress)
// {
//     // Now try and loop over any ingress adaptors for the sink
//     auto adapted_ingress = EdgeBuilder::do_adapt_ingress(target_type, ingress);

//     // Try it again in case we need a sink adaptor then a source adaptor (Short circuits if we are already there)
//     adapted_ingress = EdgeBuilder::do_adapt_ingress(target_type, adapted_ingress);

//     // Convert if neccessary
//     // auto ingress_adapted = EdgeBuilder::ingress_adapter_for_sink(source, sink, ingress);

//     // Set to the source
//     return adapted_ingress;
// }

std::shared_ptr<IngressHandleObj> EdgeBuilder::do_adapt_ingress(const EdgeTypePair& target_type,
                                                                std::shared_ptr<IngressHandleObj> ingress)
{
    // Short circuit if we are already there
    if (target_type.full_type() == ingress->get_type().full_type())
    {
        return ingress;
    }

    // Next check the static converters
    if (srf::node::EdgeRegistry::has_converter(target_type.full_type(), ingress->get_type().full_type()))
    {
        auto fn_converter =
            srf::node::EdgeRegistry::find_converter(target_type.full_type(), ingress->get_type().full_type());

        auto converted_edge = fn_converter(ingress->get_ingress());

        return std::make_shared<IngressHandleObj>(converted_edge);
    }

    // Start dynamic lookup
    VLOG(2) << "Looking for edge adapter: (" << type_name(target_type.full_type()) << ", "
            << type_name(ingress->get_type().full_type()) << ")";
    VLOG(2) << "- (" << target_type.full_type().hash_code() << ", " << ingress->get_type().full_type().hash_code()
            << ")";

    // Loop over the registered adaptors
    const auto& adaptors = EdgeAdapterRegistry::registered_ingress_adapters;

    for (const auto& adapt : adaptors)
    {
        // Try the adaptor out
        auto adapt_out = adapt(target_type, ingress->get_ingress());

        if (adapt_out)
        {
            // Check that the adaptor didnt return the same thing
            if (adapt_out->get_type().full_type() == ingress->get_type().full_type())
            {
                LOG(WARNING) << "Adaptor returned the same type as the input. Adaptors should return nullptr if the "
                                "conversion is not supported. Skipping this adaptor";
                continue;
            }

            return adapt_out;
        }
    }

    // Unfortunately, no converter was found
    throw srf::exceptions::SrfRuntimeError("No conversion found from X to Y");
}

// std::shared_ptr<channel::IEdgeWritableBase> EdgeBuilder::ingress_for_source_type(
//     std::type_index source_type, IIngressProviderBase& sink, std::shared_ptr<IEdgeWritableBase> ingress_handle)
// {
//     if (EdgeAdapterRegistry::has_sink_adapter(sink.sink_type()))
//     {
//         auto adapter = EdgeAdapterRegistry::find_sink_adapter(sink.sink_type());

//         // // Try and build the handle
//         // auto handle = adapter(source_type, sink, sink.ingress_handle());
//         // if (handle)
//         // {
//         //     return handle;
//         // }
//     }

//     // Fallback -- probably fail
//     auto fn_converter = srf::node::EdgeRegistry::find_converter(source_type, sink.sink_type());
//     return fn_converter(ingress_handle);
// }

}  // namespace srf::node

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

#include "mrc/node/edge_builder.hpp"

#include "mrc/exceptions//runtime_error.hpp"
#include "mrc/node/channel_holder.hpp"
#include "mrc/node/edge_adapter_registry.hpp"
#include "mrc/node/edge_registry.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>

namespace mrc::node {

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
    // Get the egress
    auto egress = source.get_egress_obj();

    // Set to the sink
    sink.set_egress_obj(egress);
}

std::shared_ptr<IngressHandleObj> EdgeBuilder::do_adapt_ingress(const EdgeTypePair& target_type,
                                                                std::shared_ptr<IngressHandleObj> ingress)
{
    // Short circuit if we are already there
    if (target_type.full_type() == ingress->get_type().full_type())
    {
        return ingress;
    }

    // Next check the static converters
    if (mrc::node::EdgeRegistry::has_converter(target_type.full_type(), ingress->get_type().full_type()))
    {
        try
        {
            auto fn_converter =
                mrc::node::EdgeRegistry::find_converter(target_type.full_type(), ingress->get_type().full_type());

            auto converted_edge = fn_converter(ingress->get_ingress());

            return std::make_shared<IngressHandleObj>(converted_edge);
        } catch (std::runtime_error e)
        {
            // Last attempt, check if types are the same and return ingress handle.
            if (target_type.full_type() == ingress->get_type().full_type())
            {
                return ingress;
            }

            throw e;
        }
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
    throw mrc::exceptions::MrcRuntimeError("No conversion found from X to Y");
}

}  // namespace mrc::node

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

#include "mrc/edge/edge_builder.hpp"

#include "mrc/edge/edge_adapter_registry.hpp"
#include "mrc/exceptions//runtime_error.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>

namespace mrc::edge {

void EdgeBuilder::make_edge_writable_typeless(IWritableAcceptorBase& source,
                                              IWritableProviderBase& sink,
                                              bool allow_narrowing)
{
    // Get the ingress
    auto ingress = sink.get_writable_edge_handle();

    // Set to the source
    source.set_writable_edge_handle(ingress);
}

void EdgeBuilder::make_edge_readable_typeless(IReadableProviderBase& source,
                                              IReadableAcceptorBase& sink,
                                              bool allow_narrowing)
{
    // Get the egress
    auto egress = source.get_readable_edge_handle();

    // Set to the sink
    sink.set_readable_edge_handle(egress);
}

std::shared_ptr<WritableEdgeHandle> EdgeBuilder::do_adapt_ingress(const EdgeTypeInfo& target_type,
                                                                  std::shared_ptr<WritableEdgeHandle> ingress)
{
    // Short circuit if we are already there
    if (target_type.full_type() == ingress->get_type().full_type())
    {
        return ingress;
    }

    // Next check the static converters
    if (mrc::edge::EdgeAdapterRegistry::has_ingress_converter(target_type.full_type(), ingress->get_type().full_type()))
    {
        try
        {
            auto fn_converter = mrc::edge::EdgeAdapterRegistry::find_ingress_converter(target_type.full_type(),
                                                                                       ingress->get_type().full_type());

            auto converted_edge = fn_converter(ingress->get_ingress());

            return std::make_shared<WritableEdgeHandle>(converted_edge);
        } catch (const std::runtime_error& e)
        {
            // Last attempt, check if types are the same and return ingress handle.
            if (target_type.full_type() == ingress->get_type().full_type())
            {
                return ingress;
            }

            throw e;
        }
    }

    // If static conversion failed, now try runtime conversion
    VLOG(2) << "Looking for edge adapter: (" << type_name(target_type.full_type()) << ", "
            << type_name(ingress->get_type().full_type()) << ")";
    VLOG(2) << "- (" << target_type.full_type().hash_code() << ", " << ingress->get_type().full_type().hash_code()
            << ")";

    // Loop over the registered adaptors
    const auto& adaptors = EdgeAdapterRegistry::get_ingress_adapters();

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
    throw mrc::exceptions::MrcRuntimeError(MRC_CONCAT_STR("No conversion found from "
                                                          << type_name(ingress->get_type().full_type()) << " to "
                                                          << type_name(target_type.full_type())));
}

std::shared_ptr<ReadableEdgeHandle> EdgeBuilder::do_adapt_egress(const EdgeTypeInfo& target_type,
                                                                 std::shared_ptr<ReadableEdgeHandle> egress)
{
    // Short circuit if we are already there
    if (target_type.full_type() == egress->get_type().full_type())
    {
        return egress;
    }

    // Next check the static converters
    if (mrc::edge::EdgeAdapterRegistry::has_egress_converter(egress->get_type().full_type(), target_type.full_type()))
    {
        try
        {
            auto fn_converter = mrc::edge::EdgeAdapterRegistry::find_egress_converter(egress->get_type().full_type(),
                                                                                      target_type.full_type());

            auto converted_edge = fn_converter(egress->get_egress());

            return std::make_shared<ReadableEdgeHandle>(converted_edge);
        } catch (const std::runtime_error& e)
        {
            // Last attempt, check if types are the same and return ingress handle.
            if (target_type.full_type() == egress->get_type().full_type())
            {
                return egress;
            }

            throw e;
        }
    }

    // If static conversion failed, now try runtime conversion
    VLOG(2) << "Looking for edge adapter: (" << type_name(target_type.full_type()) << ", "
            << type_name(egress->get_type().full_type()) << ")";
    VLOG(2) << "- (" << target_type.full_type().hash_code() << ", " << egress->get_type().full_type().hash_code()
            << ")";

    // Loop over the registered adaptors
    const auto& adaptors = EdgeAdapterRegistry::get_egress_adapters();

    for (const auto& adapt : adaptors)
    {
        // Try the adaptor out
        auto adapt_out = adapt(target_type, egress->get_egress());

        if (adapt_out)
        {
            // Check that the adaptor didnt return the same thing
            if (adapt_out->get_type().full_type() == egress->get_type().full_type())
            {
                LOG(WARNING) << "Adaptor returned the same type as the input. Adaptors should return nullptr if the "
                                "conversion is not supported. Skipping this adaptor";
                continue;
            }

            return adapt_out;
        }
    }

    // Unfortunately, no converter was found
    throw mrc::exceptions::MrcRuntimeError(MRC_CONCAT_STR("No conversion found from "
                                                          << type_name(egress->get_type().full_type()) << " to "
                                                          << type_name(target_type.full_type())));
}

}  // namespace mrc::edge

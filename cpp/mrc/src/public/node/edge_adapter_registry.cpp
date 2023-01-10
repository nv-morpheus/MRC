/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/node/edge_adapter_registry.hpp"

#include "mrc/utils/string_utils.hpp"
#include "mrc/utils/type_utils.hpp"

#include <map>
#include <stdexcept>
#include <typeindex>
#include <utility>

namespace mrc::node {

// Goes from source type to sink type
std::map<std::type_index, std::map<std::type_index, EdgeAdapterRegistry::ingress_converter_fn_t>>
    EdgeAdapterRegistry::registered_ingress_converters{};

std::vector<EdgeAdapterRegistry::ingress_adapter_fn_t> EdgeAdapterRegistry::registered_ingress_adapters{};

std::recursive_mutex EdgeAdapterRegistry::s_mutex{};

void EdgeAdapterRegistry::register_ingress_converter(std::type_index input_type,
                                                     std::type_index output_type,
                                                     ingress_converter_fn_t converter_fn)
{
    std::lock_guard<std::recursive_mutex> lock(s_mutex);

    VLOG(20) << "Registering converter for " << type_name(input_type) << " " << type_name(output_type);
    auto readers_map = EdgeAdapterRegistry::registered_ingress_converters[input_type];

    auto reader_found = readers_map.find(output_type);

    if (reader_found != readers_map.end())
    {
        throw std::runtime_error("Duplicate converter already registered");
    }

    EdgeAdapterRegistry::registered_ingress_converters[input_type][output_type] = converter_fn;
}

bool EdgeAdapterRegistry::has_ingress_converter(std::type_index input_type, std::type_index output_type)
{
    std::lock_guard<std::recursive_mutex> lock(s_mutex);

    auto writer_found = EdgeAdapterRegistry::registered_ingress_converters.find(input_type);

    if (writer_found == EdgeAdapterRegistry::registered_ingress_converters.end())
    {
        return false;
    }

    return writer_found->second.find(output_type) != writer_found->second.end();
}

EdgeAdapterRegistry::ingress_converter_fn_t EdgeAdapterRegistry::find_ingress_converter(std::type_index input_type,
                                                                                        std::type_index output_type)
{
    std::lock_guard<std::recursive_mutex> lock(s_mutex);

    auto writer_found = EdgeAdapterRegistry::registered_ingress_converters.find(input_type);

    if (writer_found == EdgeAdapterRegistry::registered_ingress_converters.end())
    {
        throw std::runtime_error(MRC_CONCAT_STR("Could not find input_type: " << type_name(input_type)));
    }

    auto reader_found = writer_found->second.find(output_type);

    if (reader_found == writer_found->second.end())
    {
        throw std::runtime_error("Could not find output_type");
    }

    return reader_found->second;
}

void EdgeAdapterRegistry::register_ingress_adapter(ingress_adapter_fn_t adapter_fn)
{
    std::lock_guard<std::recursive_mutex> lock(s_mutex);

    EdgeAdapterRegistry::registered_ingress_adapters.emplace_back(std::move(adapter_fn));
}

const std::vector<EdgeAdapterRegistry::ingress_adapter_fn_t>& EdgeAdapterRegistry::get_ingress_adapters()
{
    return EdgeAdapterRegistry::registered_ingress_adapters;
}

}  // namespace mrc::node

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

#include "srf/node/edge_adapter_registry.hpp"

#include <map>
#include <stdexcept>
#include <typeindex>
#include <utility>

namespace srf::node {

std::map<std::type_index, EdgeAdapterRegistry::source_adapter_fn_t> EdgeAdapterRegistry::registered_source_adapters{};
std::map<std::type_index, EdgeAdapterRegistry::sink_adapter_fn_t> EdgeAdapterRegistry::registered_sink_adapters{};

std::recursive_mutex EdgeAdapterRegistry::s_mutex{};

void EdgeAdapterRegistry::register_source_adapter(std::type_index source_type, source_adapter_fn_t adapter_fn)
{
    std::lock_guard<std::recursive_mutex> lock(s_mutex);
    auto iter_source = EdgeAdapterRegistry::registered_source_adapters.find(source_type);
    if (iter_source != EdgeAdapterRegistry::registered_source_adapters.end())
    {
        throw std::runtime_error("Duplicate edge adapter already registered");
    }

    EdgeAdapterRegistry::registered_source_adapters[source_type] = adapter_fn;
}

void EdgeAdapterRegistry::register_sink_adapter(std::type_index sink_type, sink_adapter_fn_t adapter_fn)
{
    std::lock_guard<std::recursive_mutex> lock(s_mutex);
    auto iter_sink = EdgeAdapterRegistry::registered_sink_adapters.find(sink_type);
    if (iter_sink != EdgeAdapterRegistry::registered_sink_adapters.end())
    {
        throw std::runtime_error("Duplicate edge adapter already registered");
    }

    EdgeAdapterRegistry::registered_sink_adapters[sink_type] = adapter_fn;
}

bool EdgeAdapterRegistry::has_source_adapter(std::type_index source_type)
{
    return (EdgeAdapterRegistry::registered_source_adapters.find(source_type) !=
            EdgeAdapterRegistry::registered_source_adapters.end());
}

bool EdgeAdapterRegistry::has_sink_adapter(std::type_index sink_type)
{
    return (EdgeAdapterRegistry::registered_sink_adapters.find(sink_type) !=
            EdgeAdapterRegistry::registered_sink_adapters.end());
}

EdgeAdapterRegistry::source_adapter_fn_t EdgeAdapterRegistry::find_source_adapter(std::type_index source_type)
{
    auto iter_source = EdgeAdapterRegistry::registered_source_adapters.find(source_type);
    if (iter_source == EdgeAdapterRegistry::registered_source_adapters.end())
    {
        throw std::runtime_error("Could not find adapter type");
    }

    return iter_source->second;
}

EdgeAdapterRegistry::sink_adapter_fn_t EdgeAdapterRegistry::find_sink_adapter(std::type_index sink_type)
{
    auto iter_sink = EdgeAdapterRegistry::registered_sink_adapters.find(sink_type);
    if (iter_sink == EdgeAdapterRegistry::registered_sink_adapters.end())
    {
        throw std::runtime_error("Could not find adapter type");
    }

    return iter_sink->second;
}
}  // namespace srf::node

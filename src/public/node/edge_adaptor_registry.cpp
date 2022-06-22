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

#include <map>
#include <stdexcept>
#include <typeindex>

namespace srf::node {

std::map<std::type_index, EdgeAdaptorRegistry::source_adaptor_fn_t> EdgeAdaptorRegistry::registered_source_adaptors{};
std::map<std::type_index, EdgeAdaptorRegistry::sink_adaptor_fn_t> EdgeAdaptorRegistry::registered_sink_adaptors{};

void EdgeAdaptorRegistry::register_source_adaptor(std::type_index source_type, source_adaptor_fn_t adaptor)
{
    auto iter_source = EdgeAdaptorRegistry::registered_source_adaptors.find(source_type);
    if (iter_source != EdgeAdaptorRegistry::registered_source_adaptors.end())
    {
        throw std::runtime_error("Duplicate edge adaptor already registered");
    }

    EdgeAdaptorRegistry::registered_source_adaptors[source_type] = adaptor;
}

void EdgeAdaptorRegistry::register_sink_adaptor(std::type_index sink_type, sink_adaptor_fn_t adaptor)
{
    auto iter_sink = EdgeAdaptorRegistry::registered_sink_adaptors.find(sink_type);
    if (iter_sink != EdgeAdaptorRegistry::registered_sink_adaptors.end())
    {
        throw std::runtime_error("Duplicate edge adaptor already registered");
    }

    EdgeAdaptorRegistry::registered_sink_adaptors[sink_type] = adaptor;
}

bool EdgeAdaptorRegistry::has_source_adaptor(std::type_index source_type)
{
    return (EdgeAdaptorRegistry::registered_source_adaptors.find(source_type) !=
            EdgeAdaptorRegistry::registered_source_adaptors.end());
}

bool EdgeAdaptorRegistry::has_sink_adaptor(std::type_index sink_type)
{
    return (EdgeAdaptorRegistry::registered_sink_adaptors.find(sink_type) !=
            EdgeAdaptorRegistry::registered_sink_adaptors.end());
}

EdgeAdaptorRegistry::source_adaptor_fn_t EdgeAdaptorRegistry::find_source_adaptor(std::type_index source_type)
{
    auto iter_source = EdgeAdaptorRegistry::registered_source_adaptors.find(source_type);
    if (iter_source == EdgeAdaptorRegistry::registered_source_adaptors.end())
    {
        throw std::runtime_error("Could not find adaptor type");
    }

    return iter_source->second;
}

EdgeAdaptorRegistry::sink_adaptor_fn_t EdgeAdaptorRegistry::find_sink_adaptor(std::type_index sink_type)
{
    auto iter_sink = EdgeAdaptorRegistry::registered_sink_adaptors.find(sink_type);
    if (iter_sink == EdgeAdaptorRegistry::registered_sink_adaptors.end())
    {
        throw std::runtime_error("Could not find adaptor type");
    }

    return iter_sink->second;
}
}  // namespace srf::node

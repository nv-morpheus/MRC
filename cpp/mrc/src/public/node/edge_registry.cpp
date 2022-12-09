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

#include "mrc/node/edge_registry.hpp"

#include <glog/logging.h>

#include <map>
#include <ostream>
#include <stdexcept>
#include <typeindex>
#include <utility>

namespace mrc::node {

void EdgeRegistry::register_converter(std::type_index writer_type, std::type_index reader_type, build_fn_t converter)
{
    VLOG(2) << "Registering converter for " << writer_type.hash_code() << " " << reader_type.hash_code();
    auto readers_map = EdgeRegistry::registered_converters[writer_type];

    auto reader_found = readers_map.find(reader_type);

    if (reader_found != readers_map.end())
    {
        throw std::runtime_error("Duplicate converter already registered");
    }

    EdgeRegistry::registered_converters[writer_type][reader_type] = converter;
}

bool EdgeRegistry::has_converter(std::type_index writer_type, std::type_index reader_type)
{
    auto writer_found = EdgeRegistry::registered_converters.find(writer_type);

    if (writer_found == EdgeRegistry::registered_converters.end())
    {
        return false;
    }

    return writer_found->second.find(reader_type) != writer_found->second.end();
}

EdgeRegistry::build_fn_t EdgeRegistry::find_converter(std::type_index writer_type, std::type_index reader_type)
{
    auto writer_found = EdgeRegistry::registered_converters.find(writer_type);

    if (writer_found == EdgeRegistry::registered_converters.end())
    {
        throw std::runtime_error("Could not find writer_type");
    }

    auto reader_found = writer_found->second.find(reader_type);

    if (reader_found == writer_found->second.end())
    {
        throw std::runtime_error("Could not find reader_type");
    }

    return reader_found->second;
}

// Goes from source type to sink type
std::map<std::type_index, std::map<std::type_index, EdgeRegistry::build_fn_t>> EdgeRegistry::registered_converters;

}  // namespace mrc::node

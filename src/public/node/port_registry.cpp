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

#include "srf/node/port_registry.hpp"

#include <map>
#include <stdexcept>
#include <typeindex>

namespace srf::node {

std::map<std::type_index, std::shared_ptr<PortUtil>> PortRegistry::s_registered_port_utils{};
std::map<std::string, std::type_index> PortRegistry::s_port_to_type_index{};

void PortRegistry::register_port_util(std::shared_ptr<PortUtil> util)
{
    PortRegistry::s_registered_port_utils[util->m_port_data_type] = util;
}

void PortRegistry::register_name_type_index_pair(std::string name, std::type_index type_index)
{
    s_port_to_type_index.insert({name, type_index});
}

void PortRegistry::register_name_type_index_pairs(std::vector<std::string> names,
                                                  std::vector<std::type_index> type_indices)
{
    CHECK(names.size() == type_indices.size());
    for (std::size_t idx = 0; idx < names.size(); idx++)
    {
        register_name_type_index_pair(names[idx], type_indices[idx]);
    }
}
bool PortRegistry::has_port_util(std::type_index type_index)
{
    return (PortRegistry::s_registered_port_utils.find(type_index) != PortRegistry::s_registered_port_utils.end());
}

std::shared_ptr<PortUtil> PortRegistry::find_port_util(std::type_index type_index)
{
    auto iter_util = PortRegistry::s_registered_port_utils.find(type_index);
    if (iter_util == PortRegistry::s_registered_port_utils.end())
    {
        throw std::runtime_error("No PortUtil registered for type");
    }

    return iter_util->second;
}
}  // namespace srf::node

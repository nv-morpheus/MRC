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

#include <glog/logging.h>

#include <cstddef>
#include <map>
#include <mutex>
#include <stdexcept>
#include <typeindex>
#include <utility>

namespace srf::node {

std::map<std::type_index, std::shared_ptr<PortUtil>> PortRegistry::s_registered_port_utils{};
std::map<std::string, std::type_index> PortRegistry::s_port_to_type_index{};
std::recursive_mutex PortRegistry::s_mutex{};

PortUtil::PortUtil(std::type_index type_index) : m_port_data_type(type_index) {}

std::shared_ptr<segment::ObjectProperties> PortUtil::try_cast_ingress_base_to_object(
    std::shared_ptr<segment::IngressPortBase> base)
{
    if (std::get<0>(m_ingress_casters) != nullptr)
    {
        auto obj = std::get<0>(m_ingress_casters)(base);

        if (obj != nullptr)
        {
            return obj;
        }
    }

    if (std::get<1>(m_ingress_casters) != nullptr)
    {
        return std::get<1>(m_ingress_casters)(base);
    }

    return nullptr;
}

std::shared_ptr<segment::ObjectProperties> PortUtil::try_cast_egress_base_to_object(
    std::shared_ptr<segment::EgressPortBase> base)
{
    if (std::get<0>(m_egress_casters) != nullptr)
    {
        auto obj = std::get<0>(m_egress_casters)(base);

        if (obj != nullptr)
        {
            return obj;
        }
    }

    if (std::get<1>(m_egress_casters) != nullptr)
    {
        return std::get<1>(m_egress_casters)(base);
    }

    return nullptr;
}

void PortRegistry::register_port_util(std::shared_ptr<PortUtil> util)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);

    auto iter_util = PortRegistry::s_registered_port_utils.find(util->m_port_data_type);
    if (iter_util != PortRegistry::s_registered_port_utils.end())
    {
        throw std::runtime_error("Duplicate port utility is already already registered");
    }
    PortRegistry::s_registered_port_utils[util->m_port_data_type] = util;
}

void PortRegistry::register_name_type_index_pair(std::string name, std::type_index type_index)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);
    s_port_to_type_index.insert({name, type_index});
}

void PortRegistry::register_name_type_index_pairs(std::vector<std::string> names,
                                                  std::vector<std::type_index> type_indices)
{
    CHECK(names.size() == type_indices.size());

    std::lock_guard<decltype(s_mutex)> lock(s_mutex);
    for (std::size_t idx = 0; idx < names.size(); idx++)
    {
        register_name_type_index_pair(names[idx], type_indices[idx]);
    }
}
bool PortRegistry::has_port_util(std::type_index type_index)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);
    return (PortRegistry::s_registered_port_utils.find(type_index) != PortRegistry::s_registered_port_utils.end());
}

std::shared_ptr<PortUtil> PortRegistry::find_port_util(std::type_index type_index)
{
    std::lock_guard<decltype(s_mutex)> lock(s_mutex);

    auto iter_util = PortRegistry::s_registered_port_utils.find(type_index);
    if (iter_util == PortRegistry::s_registered_port_utils.end())
    {
        throw std::runtime_error("No PortUtil registered for type");
    }

    return iter_util->second;
}
}  // namespace srf::node

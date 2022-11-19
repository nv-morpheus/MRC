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

#include "srf/segment/builder.hpp"

#include "srf/modules/module_registry.hpp"
#include "srf/node/port_registry.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace {

std::string accum_merge(std::string lhs, std::string rhs)
{
    if (lhs.empty())
    {
        return std::move(rhs);
    }

    return std::move(lhs) + "/" + std::move(rhs);
}

}  // namespace

namespace srf::segment {
std::shared_ptr<ObjectProperties> Builder::get_ingress(std::string name, std::type_index type_index)
{
    auto base = m_backend.get_ingress_base(name);
    if (!base)
    {
        throw exceptions::SrfRuntimeError("Egress port name not found: " + name);
    }

    auto port_util = node::PortRegistry::find_port_util(type_index);
    auto port      = port_util->try_cast_ingress_base_to_object(base);
    if (port == nullptr)
    {
        throw exceptions::SrfRuntimeError("Egress port type mismatch: " + name);
    }

    return port;
}

std::shared_ptr<ObjectProperties> Builder::get_egress(std::string name, std::type_index type_index)
{
    auto base = m_backend.get_egress_base(name);
    if (!base)
    {
        throw exceptions::SrfRuntimeError("Egress port name not found: " + name);
    }

    auto port_util = node::PortRegistry::find_port_util(type_index);

    auto port = port_util->try_cast_egress_base_to_object(base);
    if (port == nullptr)
    {
        throw exceptions::SrfRuntimeError("Egress port type mismatch: " + name);
    }

    return port;
}

void Builder::init_module(sp_segment_module_t module)
{
    ns_push(module);
    VLOG(2) << "Initializing module: " << m_namespace_prefix;
    module->m_module_instance_registered_namespace = m_namespace_prefix;
    module->initialize(*this);
    ns_pop();
}

std::shared_ptr<srf::modules::SegmentModule> Builder::load_module_from_registry(const std::string& module_id,
                                                                                const std::string& registry_namespace,
                                                                                std::string module_name,
                                                                                nlohmann::json config)
{
    auto fn_module_constructor = srf::modules::ModuleRegistry::get_module_constructor(module_id, registry_namespace);
    auto module                = fn_module_constructor(std::move(module_name), std::move(config));

    init_module(module);

    return module;
}

/** private implementations **/

void Builder::ns_push(sp_segment_module_t module)
{
    m_module_stack.push_back(module);
    m_namespace_stack.push_back(module->component_prefix());
    m_namespace_prefix =
        std::accumulate(m_namespace_stack.begin(), m_namespace_stack.end(), std::string(""), ::accum_merge);
}

void Builder::ns_pop()
{
    m_module_stack.pop_back();
    m_namespace_stack.pop_back();
    m_namespace_prefix =
        std::accumulate(m_namespace_stack.begin(), m_namespace_stack.end(), std::string(""), ::accum_merge);
}

void Builder::register_module_input(std::string input_name, sp_obj_prop_t object)
{
    if (m_module_stack.empty())
    {
        std::stringstream sstream;

        sstream << "Failed to register module input '" << input_name << "' -> no module context exists";
        VLOG(2) << sstream.str();

        throw std::invalid_argument(sstream.str());
    }

    auto current_module = m_module_stack.back();
    current_module->register_input_port(std::move(input_name), object);
}

void Builder::register_module_output(std::string output_name, sp_obj_prop_t object)
{
    if (m_module_stack.empty())
    {
        std::stringstream sstream;

        sstream << "Failed to register module output'" << output_name << "' -> no module context exists";
        VLOG(2) << sstream.str();

        throw std::invalid_argument(sstream.str());
    }

    auto current_module = m_module_stack.back();

    current_module->register_output_port(std::move(output_name), object);
}

}  // namespace srf::segment

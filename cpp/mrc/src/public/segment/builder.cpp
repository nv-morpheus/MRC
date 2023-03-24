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

#include "mrc/segment/builder.hpp"

#include "mrc/modules/module_registry.hpp"
#include "mrc/modules/properties/persistent.hpp"  // IWYU pragma: keep
#include "mrc/modules/segment_modules.hpp"
#include "mrc/node/port_registry.hpp"

#include <nlohmann/json.hpp>

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

namespace mrc::segment {

void Builder::init_module(std::shared_ptr<mrc::modules::SegmentModule> smodule)
{
    ns_push(smodule);
    VLOG(2) << "Initializing module: " << m_namespace_prefix;
    smodule->m_module_instance_registered_namespace = m_namespace_prefix;
    smodule->initialize(*this);
    ns_pop();

    // TODO(Devin): Maybe a better way to do this with compile time type ledger.
    if (std::dynamic_pointer_cast<modules::PersistentModule>(smodule) != nullptr)
    {
        VLOG(2) << "Registering persistent module -> '" << smodule->component_prefix() << "'";
        m_backend.add_module(m_namespace_prefix, smodule);
    }
}

std::shared_ptr<ObjectProperties> Builder::get_ingress(std::string name, std::type_index type_index)
{
    auto base = m_backend.get_ingress_base(name);
    if (!base)
    {
        throw exceptions::MrcRuntimeError("Egress port name not found: " + name);
    }

    auto port_util = node::PortRegistry::find_port_util(type_index);
    auto port      = port_util->try_cast_ingress_base_to_object(base);
    if (port == nullptr)
    {
        throw exceptions::MrcRuntimeError("Egress port type mismatch: " + name);
    }

    return port;
}

std::shared_ptr<ObjectProperties> Builder::get_egress(std::string name, std::type_index type_index)
{
    auto base = m_backend.get_egress_base(name);
    if (!base)
    {
        throw exceptions::MrcRuntimeError("Egress port name not found: " + name);
    }

    auto port_util = node::PortRegistry::find_port_util(type_index);

    auto port = port_util->try_cast_egress_base_to_object(base);
    if (port == nullptr)
    {
        throw exceptions::MrcRuntimeError("Egress port type mismatch: " + name);
    }

    return port;
}

std::shared_ptr<mrc::modules::SegmentModule> Builder::load_module_from_registry(const std::string& module_id,
                                                                                const std::string& registry_namespace,
                                                                                std::string module_name,
                                                                                nlohmann::json config)
{
    auto fn_module_constructor = mrc::modules::ModuleRegistry::get_module_constructor(module_id, registry_namespace);
    auto smodule               = fn_module_constructor(std::move(module_name), std::move(config));

    init_module(smodule);

    return smodule;
}

/** private implementations **/

void Builder::ns_push(sp_segment_module_t smodule)
{
    m_module_stack.push_back(smodule);
    m_namespace_stack.push_back(smodule->component_prefix());
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

void Builder::register_module_input(std::string input_name, std::shared_ptr<segment::ObjectProperties> object)
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

[[maybe_unused]] void Builder::register_module_output(std::string output_name,
                                                      std::shared_ptr<segment::ObjectProperties> object)
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

[[maybe_unused]] nlohmann::json Builder::get_current_module_config()
{
    if (m_module_stack.empty())
    {
        std::stringstream sstream;

        sstream << "Failed to acquire module configuration -> no module context exists";
        VLOG(2) << sstream.str();

        throw std::invalid_argument(sstream.str());
    }

    auto current_module = m_module_stack.back();

    return current_module->config();
}

}  // namespace mrc::segment

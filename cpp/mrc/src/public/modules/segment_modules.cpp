/*
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

#include "mrc/modules/segment_modules.hpp"

#include "mrc/edge/edge.hpp"
#include "mrc/edge/edge_readable.hpp"
#include "mrc/edge/edge_writable.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/utils/string_utils.hpp"

#include <glog/logging.h>

#include <ostream>
#include <stdexcept>
#include <utility>

namespace mrc::modules {

SegmentModule::SegmentModule(std::string module_name) : m_module_instance_name(std::move(module_name))
{
    if (m_module_instance_name.find_first_of("/") != std::string::npos)
    {
        throw std::invalid_argument("Module name cannot contain '/' characters");
    }
}

SegmentModule::SegmentModule(std::string module_name, nlohmann::json config) :
  m_module_instance_name(std::move(module_name)),
  m_config(std::move(config))
{
    if (m_module_instance_name.find_first_of("/") != std::string::npos)
    {
        throw std::invalid_argument("Module name cannot contain '/' characters");
    }
}

std::string SegmentModule::component_prefix() const
{
    return module_type_name() + "::" + name();
}

const nlohmann::json& SegmentModule::config() const
{
    return m_config;
}

const std::vector<std::string>& SegmentModule::input_ids() const
{
    return m_input_port_ids;
}

const SegmentModule::segment_module_port_map_t& SegmentModule::input_ports() const
{
    return m_input_ports;
}

SegmentModule::segment_module_port_t SegmentModule::input_port(const std::string& input_name)
{
    if (m_input_ports.find(input_name) != m_input_ports.end())
    {
        return m_input_ports[input_name];
    }

    std::stringstream sstream;

    sstream << "Invalid input port: " << input_name;
    throw std::invalid_argument(sstream.str());
}

const SegmentModule::segment_module_typeindex_map_t& SegmentModule::input_port_type_ids() const
{
    return m_input_port_type_indices;
}

std::type_index SegmentModule::input_port_type_id(const std::string& input_name)
{
    auto ipt_iter = m_input_port_type_indices.find(input_name);
    if (ipt_iter != m_input_port_type_indices.end())
    {
        return ipt_iter->second;
    }

    std::stringstream sstream;

    sstream << "Invalid input port: " << input_name;
    throw std::invalid_argument(sstream.str());
}

const SegmentModule::segment_module_port_map_t& SegmentModule::output_ports() const
{
    return m_output_ports;
}

SegmentModule::segment_module_port_t SegmentModule::output_port(const std::string& output_name)
{
    if (m_output_ports.find(output_name) != m_output_ports.end())
    {
        return m_output_ports[output_name];
    }

    std::stringstream sstream;

    sstream << "Invalid output port: " << output_name;
    throw std::invalid_argument(sstream.str());
}

const SegmentModule::segment_module_typeindex_map_t& SegmentModule::output_port_type_ids() const
{
    return m_output_port_type_indices;
}

std::type_index SegmentModule::output_port_type_id(const std::string& output_name)
{
    auto opt_iter = m_output_port_type_indices.find(output_name);
    if (opt_iter != m_output_port_type_indices.end())
    {
        return opt_iter->second;
    }

    std::stringstream sstream;

    sstream << "Invalid output port: " << output_name;
    throw std::invalid_argument(sstream.str());
}

const std::vector<std::string>& SegmentModule::output_ids() const
{
    return m_output_port_ids;
}

const std::string& SegmentModule::name() const
{
    return m_module_instance_name;
}

void SegmentModule::operator()(segment::IBuilder& builder)
{
    this->initialize(builder);
}

void SegmentModule::register_input_port(std::string input_name, std::shared_ptr<segment::ObjectProperties> object)
{
    // Seems to be required for ingress ports
    if (object->is_sink())
    {
        register_typed_input_port(std::move(input_name), object, object->sink_type());
        return;
    }

    if (object->is_writable_provider())
    {
        auto& writable_provider = object->writable_provider_base();
        register_typed_output_port(std::move(input_name),
                                   object,
                                   writable_provider.writable_provider_type().unwrapped_type());
        return;
    }

    if (object->is_readable_acceptor())
    {
        auto& readable_acceptor = object->readable_acceptor_base();
        register_typed_input_port(std::move(input_name),
                                  object,
                                  readable_acceptor.readable_acceptor_type().unwrapped_type());
        return;
    }

    throw std::invalid_argument("Input port object must be a writable provider or readable acceptor");
}

void SegmentModule::register_object(std::string name, std::shared_ptr<segment::ObjectProperties> object)
{
    if (!name.starts_with(MRC_CONCAT_STR(m_module_instance_registered_namespace << "/")))
    {
        throw std::invalid_argument(MRC_CONCAT_STR("Attempt to register object with invalid name: "
                                                   << name
                                                   << " for module: " << m_module_instance_registered_namespace));
    }

    auto local_name = name.substr(m_module_instance_registered_namespace.size() + 1);

    if (m_objects.find(local_name) != m_objects.end())
    {
        throw std::invalid_argument(MRC_CONCAT_STR("Attempt to register duplicate module object: " << std::move(name)));
    }

    m_objects[local_name] = object;
}

segment::ObjectProperties& SegmentModule::find_object(const std::string& name) const
{
    if (!name.starts_with(MRC_CONCAT_STR(m_module_instance_registered_namespace << "/")))
    {
        throw std::invalid_argument(MRC_CONCAT_STR("Attempt to find object with invalid name: "
                                                   << name
                                                   << " for module: " << m_module_instance_registered_namespace));
    }

    auto local_name = name.substr(m_module_instance_registered_namespace.size() + 1);

    auto found = m_objects.find(local_name);

    if (found == m_objects.end())
    {
        throw exceptions::MrcRuntimeError(MRC_CONCAT_STR("Unable to find segment object with name "
                                                         << name << " in module "
                                                         << m_module_instance_registered_namespace));
    }

    return *found->second;
}

void SegmentModule::register_typed_input_port(std::string input_name,
                                              std::shared_ptr<segment::ObjectProperties> object,
                                              std::type_index tidx)
{
    if (m_input_ports.find(input_name) != m_input_ports.end())
    {
        std::stringstream sstream;

        sstream << "Attempt to register duplicate module input port: " + std::move(input_name);
        throw std::invalid_argument(sstream.str());
    }

    VLOG(5) << "Registering input port: " << input_name << " with type: " << tidx.name() << " for module: " << name();

    m_input_port_ids.push_back(input_name);
    m_input_ports[input_name] = object;
    m_input_port_type_indices.try_emplace(input_name, tidx);
}

void SegmentModule::register_output_port(std::string output_name, std::shared_ptr<segment::ObjectProperties> object)
{
    // Seems to be necessary for egress ports.
    if (object->is_source())
    {
        register_typed_output_port(std::move(output_name), object, object->source_type());
        return;
    }

    if (object->is_writable_acceptor())
    {
        auto& writable_acceptor = object->writable_acceptor_base();
        register_typed_output_port(std::move(output_name),
                                   object,
                                   writable_acceptor.writable_acceptor_type().unwrapped_type());
        return;
    }

    if (object->is_readable_provider())
    {
        auto& readable_provider = object->readable_provider_base();
        register_typed_output_port(std::move(output_name),
                                   object,
                                   readable_provider.readable_provider_type().unwrapped_type());
        return;
    }

    throw std::invalid_argument("Output port object must be a writable acceptor or readable provider");
}

void SegmentModule::register_typed_output_port(std::string output_name,
                                               std::shared_ptr<segment::ObjectProperties> object,
                                               std::type_index tidx)
{
    if (m_output_ports.find(output_name) != m_output_ports.end())
    {
        std::stringstream sstream;

        sstream << "Attempt to register duplicate module output port: " + std::move(output_name);
        throw std::invalid_argument(sstream.str());
    }

    VLOG(5) << "Registering output port: " << output_name << " with type: " << tidx.name() << " for module: " << name();

    m_output_port_ids.push_back(output_name);
    m_output_ports[output_name] = object;
    m_output_port_type_indices.try_emplace(output_name, tidx);
}

}  // namespace mrc::modules

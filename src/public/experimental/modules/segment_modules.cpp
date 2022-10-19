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

#include "srf/experimental/modules/segment_modules.hpp"

namespace srf::modules {

SegmentModule::SegmentModule(std::string module_name) : m_module_name(std::move(module_name))
{
    std::stringstream sstream;

    sstream << "segment_module/" << m_module_name << "/";
    m_component_prefix = sstream.str();
}

SegmentModule::SegmentModule(std::string module_name, nlohmann::json config) :
  m_module_name(std::move(module_name)),
  m_config(std::move(config))
{
    std::stringstream sstream;

    sstream << "segment_module/" << m_module_name << "/";
    m_component_prefix = sstream.str();
}

const std::string& SegmentModule::component_prefix() const
{
    return m_component_prefix;
}

const nlohmann::json& SegmentModule::config() const
{
    return m_config;
}

std::string SegmentModule::get_module_component_name(const std::string& component_name) const
{
    return component_prefix() + component_name;
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

const std::map<std::string, const std::type_info*>& SegmentModule::input_port_type_ids() const
{
    return m_input_port_type_indices;
}

const std::type_info* SegmentModule::input_port_type_id(const std::string& input_name)
{
    if (m_input_port_type_indices.find(input_name) != m_input_port_type_indices.end())
    {
        return m_input_port_type_indices[input_name];
    }

    std::stringstream sstream;

    sstream << "Invalid input port: " << input_name;

    throw std::invalid_argument(sstream.str());
}

const SegmentModule::segment_module_port_map_t SegmentModule::output_ports() const
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

const SegmentModule::segment_module_typeinfo_map_t& SegmentModule::output_port_type_ids() const
{
    return m_output_port_type_indices;
}

const std::type_info* SegmentModule::output_port_type_id(const std::string& output_name)
{
    if (m_output_port_type_indices.find(output_name) != m_output_port_type_indices.end())
    {
        return m_output_port_type_indices[output_name];
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
    return m_module_name;
}

void SegmentModule::register_input_port(std::string input_name,
                                        std::shared_ptr<segment::ObjectProperties> object,
                                        const std::type_info* tinfo)
{
    m_input_port_ids.push_back(input_name);
    m_input_ports[input_name]             = object;
    m_input_port_type_indices[input_name] = tinfo;
}

void SegmentModule::register_output_port(std::string output_name,
                                         std::shared_ptr<segment::ObjectProperties> object,
                                         const std::type_info* tinfo)
{
    m_output_port_ids.push_back(output_name);
    m_output_ports[output_name]             = object;
    m_output_port_type_indices[output_name] = tinfo;
}

}  // namespace srf::modules

/*
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

#include "internal/segment/builder_definition.hpp"

#include "internal/pipeline/pipeline_resources.hpp"
#include "internal/segment/segment_definition.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/metrics/counter.hpp"
#include "mrc/metrics/registry.hpp"
#include "mrc/modules/module_registry.hpp"
#include "mrc/modules/properties/persistent.hpp"  // IWYU pragma: keep
#include "mrc/modules/segment_modules.hpp"
#include "mrc/node/port_registry.hpp"
#include "mrc/runnable/launchable.hpp"
#include "mrc/segment/egress_port.hpp"   // IWYU pragma: keep
#include "mrc/segment/ingress_port.hpp"  // IWYU pragma: keep
#include "mrc/segment/object.hpp"
#include "mrc/types.hpp"

#include <glog/logging.h>

#include <exception>
#include <memory>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace {

std::string accum_merge(std::string lhs, std::string rhs)
{
    if (lhs.empty())
    {
        return std::move(rhs);
    }

    return std::move(lhs) + "/" + std::move(rhs);
}

/**
 * @brief Checks for invalid characters in a name. Throws an exception if invalid characters are found.
 *
 * @param name The name to check
 */
void validate_name(const std::string& name)
{
    auto found_bad_characters = name.find_first_of("/");

    if (found_bad_characters != std::string::npos)
    {
        throw mrc::exceptions::MrcRuntimeError("Invalid name `" + name +
                                               "'. Cannot contain any of these characters: '/'");
    }
}

}  // namespace

namespace mrc::segment {

BuilderDefinition::BuilderDefinition(std::shared_ptr<const SegmentDefinition> definition,
                                     SegmentRank rank,
                                     pipeline::PipelineResources& resources,
                                     std::size_t default_partition_id) :
  m_definition(std::move(definition)),
  m_rank(rank),
  m_resources(resources),
  m_default_partition_id(default_partition_id)
{}

std::shared_ptr<BuilderDefinition> BuilderDefinition::unwrap(std::shared_ptr<IBuilder> object)
{
    // Convert to the full implementation
    auto full_object = std::dynamic_pointer_cast<BuilderDefinition>(object);

    CHECK(full_object) << "Invalid cast for BuilderDefinition. Please report to the developers";

    return full_object;
}

const std::string& BuilderDefinition::name() const
{
    return m_definition->name();
}

std::tuple<std::string, std::string, std::string> BuilderDefinition::normalize_name(const std::string& name,
                                                                                    bool ignore_namespace) const
{
    // Prefix all nodes with `/<SegmentName>/`
    auto global_prefix = "/" + this->name() + "/";

    // Check and see if the name starts with "/" which means its global
    bool is_global = name.starts_with(global_prefix);

    std::string local_name;
    std::string child_name;

    if (is_global)
    {
        // Local is everything after the global prefix
        local_name = name.substr(global_prefix.length());
    }
    else
    {
        // Otherwise build up the local name from any module prefix
        local_name = (ignore_namespace || m_namespace_prefix.empty()) ? name : m_namespace_prefix + "/" + name;
    }

    auto global_name = global_prefix + local_name;

    // Check for a child name
    auto child_name_start_idx = local_name.find("/");

    if (child_name_start_idx != std::string::npos)
    {
        child_name = local_name.substr(child_name_start_idx + 1);
        local_name = local_name.substr(0, child_name_start_idx);
    }

    return std::make_tuple(global_name, local_name, child_name);
}

std::shared_ptr<ObjectProperties> BuilderDefinition::get_ingress(std::string name, std::type_index type_index)
{
    auto base = this->get_ingress_base(name);
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

std::shared_ptr<ObjectProperties> BuilderDefinition::get_egress(std::string name, std::type_index type_index)
{
    auto base = this->get_egress_base(name);
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

void BuilderDefinition::init_module(std::shared_ptr<mrc::modules::SegmentModule> smodule)
{
    this->module_push(smodule);
    VLOG(2) << "Initializing module: " << m_namespace_prefix;
    smodule->m_module_instance_registered_namespace = m_namespace_prefix;
    smodule->initialize(*this);

    // TODO(Devin): Maybe a better way to do this with compile time type ledger.
    if (auto persist = std::dynamic_pointer_cast<modules::PersistentModule>(smodule))
    {
        VLOG(2) << "Registering persistent module -> '" << m_namespace_prefix << "'";

        // Just save to a vector to keep it alive
        m_modules.push_back(persist);
    }

    this->module_pop(smodule);
}

void BuilderDefinition::register_module_input(std::string input_name, std::shared_ptr<segment::ObjectProperties> object)
{
    validate_name(input_name);

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

[[maybe_unused]] nlohmann::json BuilderDefinition::get_current_module_config()
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

[[maybe_unused]] void BuilderDefinition::register_module_output(std::string output_name,
                                                                std::shared_ptr<segment::ObjectProperties> object)
{
    validate_name(output_name);

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

std::shared_ptr<mrc::modules::SegmentModule> BuilderDefinition::load_module_from_registry(
    const std::string& module_id,
    const std::string& registry_namespace,
    std::string module_name,
    nlohmann::json config)
{
    auto fn_module_constructor = mrc::modules::ModuleRegistry::get_module_constructor(module_id, registry_namespace);
    auto smodule               = fn_module_constructor(std::move(module_name), std::move(config));

    init_module(smodule);

    return smodule;
}

const SegmentDefinition& BuilderDefinition::definition() const
{
    return *m_definition;
}

void BuilderDefinition::initialize()
{
    auto address = segment_address_encode(this->definition().id(), m_rank);

    // construct ingress ports
    for (const auto& [name, initializer] : this->definition().ingress_initializers())
    {
        DVLOG(10) << "constructing ingress_port: " << name;
        auto port = initializer(address);
        this->add_object(name, port);
    }

    // construct egress ports
    for (const auto& [name, initializer] : this->definition().egress_initializers())
    {
        DVLOG(10) << "constructing egress_port: " << name;
        auto port = initializer(address);
        this->add_object(name, port);
    }

    // Call the segment initializer
    try
    {
        m_definition->initializer_fn()(*this);
    } catch (const std::exception& e)
    {
        LOG(ERROR) << "Exception during segment initializer. Segment name: " << m_definition->name()
                   << ", Segment Rank: " << m_rank << ". Exception message:\n"
                   << e.what();

        // Rethrow after logging
        std::rethrow_exception(std::current_exception());
    }
}

const std::map<std::string, std::shared_ptr<mrc::runnable::Launchable>>& BuilderDefinition::nodes() const
{
    return m_nodes;
}

const std::map<std::string, std::shared_ptr<::mrc::segment::EgressPortBase>>& BuilderDefinition::egress_ports() const
{
    return m_egress_ports;
}

const std::map<std::string, std::shared_ptr<::mrc::segment::IngressPortBase>>& BuilderDefinition::ingress_ports() const
{
    return m_ingress_ports;
}

bool BuilderDefinition::has_object(const std::string& name) const
{
    auto [global_name, local_name, child_name] = this->normalize_name(name);

    auto search = m_objects.find(local_name);
    if (search == m_objects.end())
    {
        return false;
    }

    if (child_name.empty())
    {
        return true;
    }

    return search->second->has_child(child_name);
}

mrc::segment::ObjectProperties& BuilderDefinition::find_object(const std::string& name)
{
    auto [global_name, local_name, child_name] = this->normalize_name(name);

    auto search = m_objects.find(local_name);
    if (search == m_objects.end())
    {
        LOG(ERROR) << "Unable to find segment object with name: " << name;
        throw exceptions::MrcRuntimeError("unable to find segment object with name " + name);
    }

    if (child_name.empty())
    {
        return *(search->second);
    }
    auto found_child = search->second->get_child(child_name);

    if (!found_child)
    {
        throw exceptions::MrcRuntimeError(
            MRC_CONCAT_STR("Unable to find segment object with name: " << name << ". Child not found: " << child_name));
    }

    return *found_child;
}

void BuilderDefinition::add_object(const std::string& name, std::shared_ptr<::mrc::segment::ObjectProperties> object)
{
    // First, ensure that the name is properly formatted
    validate_name(name);

    if (has_object(name))
    {
        LOG(ERROR) << "A Object named " << name << " is already registered";
        throw exceptions::MrcRuntimeError("duplicate name detected - name owned by a node");
    }

    auto [global_name, local_name, child_name] = this->normalize_name(name);

    if (!child_name.empty())
    {
        throw exceptions::MrcRuntimeError(
            MRC_CONCAT_STR("Object names cannot contain '/' characters. Found: " << local_name << "/" << child_name));
    }

    m_objects[local_name] = object;

    // Now initialize the object with the name (this will generate children as well)
    object->initialize(global_name, this);

    if (object->is_runnable())
    {
        auto launchable = std::dynamic_pointer_cast<runnable::Launchable>(object);

        CHECK(launchable) << "Invalid conversion. Object returned is_runnable() == true, but was not of type "
                             "Launchable";

        m_nodes[local_name] = launchable;
    }

    // Add to ingress ports list if it is the right type
    if (auto ingress_port = std::dynamic_pointer_cast<IngressPortBase>(object))
    {
        // Save by the original name
        m_ingress_ports[local_name] = ingress_port;
    }

    // Add to egress ports list if it is the right type
    if (auto egress_port = std::dynamic_pointer_cast<EgressPortBase>(object))
    {
        // Save by the original name
        m_egress_ports[local_name] = egress_port;
    }

    // // Now register any child objects
    // auto children = object->get_children();

    // if (!children.empty())
    // {
    //     // Push the namespace for this object
    //     this->ns_push(local_name);

    //     for (auto& [child_name, child_object] : children)
    //     {
    //         // Add the child object
    //         this->add_object(child_name, child_object);
    //     }

    //     // Pop the namespace for this object
    //     this->ns_pop(local_name);
    // }
}

std::shared_ptr<::mrc::segment::IngressPortBase> BuilderDefinition::get_ingress_base(const std::string& name)
{
    auto [global_name, local_name, child_name] = this->normalize_name(name, true);

    DCHECK_EQ(child_name, "") << "Child names are not supported for ingress ports. Found: " << local_name << "/"
                              << child_name;

    auto search = m_ingress_ports.find(local_name);

    if (search != m_ingress_ports.end())
    {
        return search->second;
    }

    return nullptr;
}

std::shared_ptr<::mrc::segment::EgressPortBase> BuilderDefinition::get_egress_base(const std::string& name)
{
    auto [global_name, local_name, child_name] = this->normalize_name(name, true);

    DCHECK_EQ(child_name, "") << "Child names are not supported for egress ports. Found: " << local_name << "/"
                              << child_name;

    auto search = m_egress_ports.find(local_name);

    if (search != m_egress_ports.end())
    {
        return search->second;
    }

    return nullptr;
}

std::function<void(std::int64_t)> BuilderDefinition::make_throughput_counter(const std::string& name)
{
    auto [global_name, local_name, child_name] = this->normalize_name(name);

    DCHECK_EQ(child_name, "") << "Child names are not supported for throughput counters ports. Found: " << local_name
                              << "/" << child_name;

    auto counter = m_resources.metrics_registry().make_throughput_counter(global_name);

    return [counter](std::int64_t ticks) mutable {
        counter.increment(ticks);
    };
}

std::string BuilderDefinition::module_push(std::shared_ptr<mrc::modules::SegmentModule> smodule)
{
    m_module_stack.push_back(smodule);

    return this->ns_push(smodule->component_prefix());
}

std::string BuilderDefinition::module_pop(std::shared_ptr<mrc::modules::SegmentModule> smodule)
{
    CHECK_EQ(smodule, m_module_stack.back())
        << "Namespace stack mismatch. Expected " << m_module_stack.back()->component_prefix() << " but got "
        << smodule->component_prefix();

    m_module_stack.pop_back();

    return this->ns_pop(smodule->component_prefix());
}

std::string BuilderDefinition::ns_push(const std::string& name)
{
    m_namespace_stack.push_back(name);
    m_namespace_prefix =
        std::accumulate(m_namespace_stack.begin(), m_namespace_stack.end(), std::string(""), ::accum_merge);

    return m_namespace_prefix;
}

std::string BuilderDefinition::ns_pop(const std::string& name)
{
    CHECK_EQ(name, m_namespace_stack.back())
        << "Namespace stack mismatch. Expected " << m_namespace_stack.back() << " but got " << name;

    m_namespace_stack.pop_back();
    m_namespace_prefix =
        std::accumulate(m_namespace_stack.begin(), m_namespace_stack.end(), std::string(""), ::accum_merge);

    return m_namespace_prefix;
}

}  // namespace mrc::segment

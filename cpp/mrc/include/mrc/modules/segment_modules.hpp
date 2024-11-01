/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <nlohmann/json.hpp>

#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <vector>

namespace mrc::segment {
class IBuilder;
class BuilderDefinition;
}  // namespace mrc::segment

namespace mrc::segment {
struct ObjectProperties;
}  // namespace mrc::segment

namespace mrc::modules {

class SegmentModule
{
  public:
    using segment_module_port_map_t      = std::map<std::string, std::shared_ptr<segment::ObjectProperties>>;
    using segment_module_port_t          = std::shared_ptr<segment::ObjectProperties>;
    using segment_module_typeindex_map_t = std::map<std::string, const std::type_index>;

    virtual ~SegmentModule() = default;

    SegmentModule() = delete;

    SegmentModule(std::string module_name);
    SegmentModule(std::string module_name, nlohmann::json config);

    std::string component_prefix() const;
    const nlohmann::json& config() const;
    const std::string& name() const;

    /**
     * Return vector of input ids -- these are only understood by the SegmentModule
     * @return std::vector
     */
    const std::vector<std::string>& input_ids() const;

    /**
     * Return a vector of output ids -- these are only understood by the SegmentModule class
     * @return std::vector
     */
    const std::vector<std::string>& output_ids() const;

    /**
     * Return a set of ObjectProperties for module input_ids
     * @return ObjectProperties
     */
    const segment_module_port_map_t& input_ports() const;

    /**
     * Return the ObjectProperties object corresponding to input_name
     * @param input_name Name of the module port
     * @return ObjectProperties
     */
    segment_module_port_t input_port(const std::string& input_name);

    /**
     * Return a map of module port id : type indices
     * @return std::map
     */
    const segment_module_typeindex_map_t& input_port_type_ids() const;

    /**
     * Return the type index of a given input name
     * @return Type index
     */
    std::type_index input_port_type_id(const std::string& input_name);

    /**
     * Return a set of ObjectProperties for module input_ids
     * @return ObjectProperties
     */
    const segment_module_port_map_t& output_ports() const;

    /**
     * Return an ObjectProperties for the module port corresponding to output name
     * @param output_name Name of the module port to return
     * @return ObjectProperties
     */
    segment_module_port_t output_port(const std::string& output_name);

    /**
     * Return a map of module port id : type indices
     * @return std::map
     */
    const segment_module_typeindex_map_t& output_port_type_ids() const;

    /**
     * Return the type index of a given input name
     * @param input_name Name of the module port to return a type index for
     * @return Type index
     */
    std::type_index output_port_type_id(const std::string& output_name);

    /**
     * Functional entrypoint for module constructor during build -- this lets us act like a std::function
     * @param builder
     */
    void operator()(segment::IBuilder& builder);

    /**
     * Retrieve the class name for the module, defaults to 'segment_module'
     * @return
     */
    virtual std::string module_type_name() const = 0;

  protected:
    // Derived class interface functions
    /* Virtual Functions */
    /**
     * Entrypoint for module constructor during build
     * @param builder
     */
    virtual void initialize(segment::IBuilder& builder) = 0;

  private:
    /**
     * @brief Registers an object with the module to keep it alive
     *
     * @param name The name of the object
     * @param object The object to register
     */
    void register_object(std::string name, std::shared_ptr<segment::ObjectProperties> object);

    /**
     * @brief Find an object by name. Must be registered with the module
     *
     * @param name The name of the object
     * @return segment::ObjectProperties&
     */
    segment::ObjectProperties& find_object(const std::string& name) const;

    /* Interface Functions */
    /**
     * Register an input port that should be exposed for the module
     * @param input_name Port name
     * @param object ObjectProperties object associated with the port
     */
    void register_input_port(std::string input_name, std::shared_ptr<segment::ObjectProperties> object);

    /**
     * Register an output port that should be exposed for the module
     * @param input_name Port name
     * @param object ObjectProperties object associated with the port
     */
    void register_output_port(std::string output_name, std::shared_ptr<segment::ObjectProperties> object);

    /**
     * Register an input port that should be exposed for the module, with explicit type index. This is
     * necessary for Objects that aren't explicit Source or Sink types (e.g. a custom object type)
     * @param input_name Port name
     * @param object ObjectProperties object associated with the port
     * @param tidx Type index of the object's payload data type
     */
    void register_typed_input_port(std::string input_name,
                                   std::shared_ptr<segment::ObjectProperties> object,
                                   std::type_index tidx);
    /**
     * Register an output port that should be exposed for the module, with explicit type index. This is
     * necessary for Objects that aren't explicit Source or Sink types (e.g. a custom object type)
     * @param output_name Port name
     * @param object ObjectProperties object associated with the port
     * @param tidx Type index of the object's payload data type
     */
    void register_typed_output_port(std::string output_name,
                                    std::shared_ptr<segment::ObjectProperties> object,
                                    std::type_index tidx);

    const std::string m_module_instance_name;

    std::string m_module_instance_registered_namespace{};

    std::vector<std::string> m_input_port_ids{};
    std::vector<std::string> m_output_port_ids{};

    segment_module_typeindex_map_t m_input_port_type_indices{};
    segment_module_typeindex_map_t m_output_port_type_indices{};

    segment_module_port_map_t m_input_ports{};
    segment_module_port_map_t m_output_ports{};

    // Maintain a map of all objects to keep them alive. These are registered as internal names
    std::map<std::string, std::shared_ptr<segment::ObjectProperties>> m_objects;

    const nlohmann::json m_config;

    friend class segment::BuilderDefinition;
};

}  // namespace mrc::modules

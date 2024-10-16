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

#pragma once

#include "mrc/segment/builder.hpp"
#include "mrc/types.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <typeindex>
#include <vector>

namespace mrc::pipeline {
class PipelineResources;
}  // namespace mrc::pipeline

namespace mrc::modules {
class SegmentModule;
class PersistentModule;
}  // namespace mrc::modules

namespace mrc::runnable {
struct Launchable;
}  // namespace mrc::runnable

namespace mrc::segment {
class EgressPortBase;
struct IngressPortBase;
struct ObjectProperties;
}  // namespace mrc::segment

namespace mrc::segment {
class SegmentDefinition;

class BuilderDefinition : public IBuilder
{
  public:
    BuilderDefinition(std::shared_ptr<const SegmentDefinition> definition,
                      SegmentRank rank,
                      pipeline::PipelineResources& resources,
                      std::size_t default_partition_id);

    static std::shared_ptr<BuilderDefinition> unwrap(std::shared_ptr<IBuilder> object);

    const std::string& name() const override;

    std::tuple<std::string, std::string> normalize_name(const std::string& name,
                                                        bool ignore_namespace = false) const override;

    std::shared_ptr<ObjectProperties> get_ingress(std::string name, std::type_index type_index) override;

    std::shared_ptr<ObjectProperties> get_egress(std::string name, std::type_index type_index) override;

    /**
     * Initialize a SegmentModule that was instantiated outside of the builder.
     * @param module Module to initialize
     */
    void init_module(std::shared_ptr<mrc::modules::SegmentModule> smodule) override;

    /**
     * Register an input port on the given module -- note: this in generally only necessary for dynamically
     * created modules that use an alternate initializer function independent of the derived class.
     * See: PythonSegmentModule
     * @param input_name Unique name of the input port
     * @param object shared pointer to type erased Object associated with 'input_name' on this module instance.
     */
    void register_module_input(std::string input_name, std::shared_ptr<ObjectProperties> object) override;

    /**
     * Get the json configuration for the current module under configuration.
     * @return nlohmann::json object.
     */
    nlohmann::json get_current_module_config() override;

    /**
     * Register an output port on the given module -- note: this in generally only necessary for dynamically
     * created modules that use an alternate initializer function independent of the derived class.
     * See: PythonSegmentModule
     * @param output_name Unique name of the output port
     * @param object shared pointer to type erased Object associated with 'output_name' on this module instance.
     */
    void register_module_output(std::string output_name, std::shared_ptr<ObjectProperties> object) override;

    /**
     * Load an existing, registered module, initialize it, and return it to the caller
     * @param module_id Unique ID of the module to load
     * @param registry_namespace Namespace where the module id is registered
     * @param module_name Unique name of this instance of the module
     * @param config Configuration to pass to the module
     * @return Return a shared pointer to the new module, which is a derived class of SegmentModule
     */
    std::shared_ptr<mrc::modules::SegmentModule> load_module_from_registry(const std::string& module_id,
                                                                           const std::string& registry_namespace,
                                                                           std::string module_name,
                                                                           nlohmann::json config = {}) override;

    const SegmentDefinition& definition() const;

    void initialize();
    void shutdown();

    const std::map<std::string, std::shared_ptr<runnable::Launchable>>& nodes() const;
    const std::map<std::string, std::shared_ptr<EgressPortBase>>& egress_ports() const;
    const std::map<std::string, std::shared_ptr<IngressPortBase>>& ingress_ports() const;

  private:
    // Overriding methods
    ObjectProperties& find_object(const std::string& name) override;
    void add_object(const std::string& name, std::shared_ptr<ObjectProperties> object) override;
    std::shared_ptr<IngressPortBase> get_ingress_base(const std::string& name) override;
    std::shared_ptr<EgressPortBase> get_egress_base(const std::string& name) override;
    std::function<void(std::int64_t)> make_throughput_counter(const std::string& name) override;

    // Local methods
    bool has_object(const std::string& name) const;

    void ns_push(std::shared_ptr<mrc::modules::SegmentModule> smodule);
    void ns_pop();

    // definition
    std::shared_ptr<const SegmentDefinition> m_definition;
    SegmentRank m_rank;

    // Resource info
    pipeline::PipelineResources& m_resources;
    const std::size_t m_default_partition_id;

    // Module info
    std::string m_namespace_prefix;
    std::vector<std::string> m_namespace_stack{};
    std::vector<std::shared_ptr<mrc::modules::SegmentModule>> m_module_stack{};

    // all objects - ports, runnables, etc.
    std::map<std::string, std::shared_ptr<ObjectProperties>> m_objects;

    // Saved modules to guarantee lifetime
    std::vector<std::shared_ptr<modules::PersistentModule>> m_modules;

    // only runnables
    std::map<std::string, std::shared_ptr<mrc::runnable::Launchable>> m_nodes;

    // ingress/egress - these are also nodes/objects
    std::map<std::string, std::shared_ptr<IngressPortBase>> m_ingress_ports;
    std::map<std::string, std::shared_ptr<EgressPortBase>> m_egress_ports;
};

}  // namespace mrc::segment

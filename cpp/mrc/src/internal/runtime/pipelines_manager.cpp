/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/runtime/pipelines_manager.hpp"

#include "internal/async_service.hpp"
#include "internal/control_plane/state/root_state.hpp"
#include "internal/pipeline/pipeline.hpp"
#include "internal/pipeline/pipeline_instance.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/runtime/runtime.hpp"
#include "internal/segment/definition.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/protos/architect.pb.h"
#include "mrc/protos/architect_state.pb.h"

#include <memory>

namespace mrc::internal::runtime {

PipelinesManager::PipelinesManager(Runtime& system_runtime) :
  AsyncService("PipelinesManager"),
  runnable::RunnableResourcesProvider(system_runtime),
  m_system_runtime(system_runtime)
{}

PipelinesManager::~PipelinesManager() = default;

void PipelinesManager::register_defs(std::vector<std::shared_ptr<pipeline::Pipeline>> pipeline_defs)
{
    // Now loop over all and register with the control plane
    for (const auto& pipeline : pipeline_defs)
    {
        auto request = protos::PipelineRequestAssignmentRequest();

        auto* config  = request.mutable_pipeline();
        auto* mapping = request.mutable_mapping();

        for (const auto& [segment_id, segment] : pipeline->segments())
        {
            protos::PipelineConfiguration_SegmentConfiguration seg_config;

            seg_config.set_name(segment->name());

            for (const auto& egress_port_name : segment->egress_port_names())
            {
                auto* egress = seg_config.mutable_egress_ports()->Add();
                egress->set_name(egress_port_name);
            }

            for (const auto& ingress_port_name : segment->ingress_port_names())
            {
                auto* ingress = seg_config.mutable_ingress_ports()->Add();
                ingress->set_name(ingress_port_name);
            }

            config->mutable_segments()->emplace(segment->name(), std::move(seg_config));

            protos::PipelineMapping_SegmentMapping seg_mapping;

            seg_mapping.set_segment_name(segment->name());

            seg_mapping.mutable_by_policy()->set_value(::mrc::protos::SegmentMappingPolicies::OnePerWorker);

            mapping->mutable_segments()->emplace(segment->name(), std::move(seg_mapping));
        }

        auto response = m_system_runtime.control_plane().await_unary<protos::PipelineRequestAssignmentResponse>(
            protos::EventType::ClientUnaryRequestPipelineAssignment,
            request);

        m_definitions[response->pipeline_definition_id()] = pipeline;
    }
}

pipeline::Pipeline& PipelinesManager::get_definition(uint64_t definition_id)
{
    CHECK(m_definitions.contains(definition_id))
        << "Pipeline with ID: " << definition_id << " not found in registered pipeline definitions";

    return *m_definitions[definition_id];
}

pipeline::PipelineInstance& PipelinesManager::get_instance(uint64_t instance_id)
{
    CHECK(m_definitions.contains(instance_id))
        << "Pipeline with ID: " << instance_id << " not found in pipeline instances";

    return *m_instances[instance_id];
}

void PipelinesManager::do_service_start(std::stop_token stop_token)
{
    Promise<void> completed_promise;

    // Now, subscribe to the control plane state updates and filter only on updates to this instance ID
    m_system_runtime.control_plane()
        .state_update_obs()
        .tap([](const control_plane::state::ControlPlaneState& state) {
            VLOG(10) << "State Update: PipelinesManager";
        })
        .subscribe(
            [this](control_plane::state::ControlPlaneState state) {
                // Handle updates to the worker
                this->process_state_update(state);
            },
            [this](std::exception_ptr ex_ptr) {
                try
                {
                    std::rethrow_exception(ex_ptr);
                } catch (std::exception ex)
                {
                    LOG(ERROR) << "Error in " << this->debug_prefix() << ex.what();
                }
            },
            [&completed_promise] {
                completed_promise.set_value();
            });

    // Need to mark this started before waiting
    this->mark_started();

    // Yield until the observable is finished
    completed_promise.get_future().wait();
}

void PipelinesManager::process_state_update(control_plane::state::ControlPlaneState& state)
{
    // Loop over all pipeline instances
    for (const auto& [pipe_instance_id, pipe_instance] : state.pipeline_instances())
    {
        // TODO(MDD): Need to filter based on our machine ID

        // Check to see if this was newly created
        if (pipe_instance.state().status() == control_plane::state::ResourceStatus::Registered)
        {
            // Get the definition for this instance
            auto def_id = pipe_instance.definition().id();

            CHECK(m_definitions.contains(def_id)) << "Unknown definition ID: " << def_id;

            auto definition = m_definitions[def_id];

            // First, double check if this still needs to be created by trying to activate it
            auto request = protos::ResourceUpdateStatusRequest();

            request.set_resource_type("PipelineInstances");
            request.set_resource_id(pipe_instance_id);
            request.set_status(protos::ResourceStatus::Activated);

            auto response = m_system_runtime.control_plane().await_unary<protos::ResourceUpdateStatusResponse>(
                protos::EventType::ClientUnaryResourceUpdateStatus,
                request);

            // Resource was activated, lets now create the object
            auto [added_iterator, did_add] = m_instances.emplace(
                pipe_instance_id,
                std::make_unique<pipeline::PipelineInstance>(m_system_runtime, definition, pipe_instance_id));

            // Now start as a child service
            this->child_service_start(*added_iterator->second);
        }
    }
}

}  // namespace mrc::internal::runtime

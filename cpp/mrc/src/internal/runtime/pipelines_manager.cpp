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

#include "internal/control_plane/client.hpp"
#include "internal/control_plane/state/root_state.hpp"
#include "internal/pipeline/pipeline_definition.hpp"
#include "internal/pipeline/pipeline_instance.hpp"
#include "internal/segment/segment_definition.hpp"

#include "mrc/core/async_service.hpp"
#include "mrc/core/error.hpp"
#include "mrc/core/utils.hpp"
#include "mrc/protos/architect.pb.h"
#include "mrc/protos/architect_state.pb.h"
#include "mrc/segment/ports.hpp"
#include "mrc/utils/ranges.hpp"
#include "mrc/utils/type_utils.hpp"

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>

#include <chrono>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <typeindex>
#include <utility>

namespace mrc::runtime {

PipelinesManager::PipelinesManager(IInternalRuntimeProvider& runtime) :
  AsyncService(MRC_CONCAT_STR("PipelinesManager")),
  InternalRuntimeProvider(runtime)
{}

PipelinesManager::~PipelinesManager()
{
    AsyncService::call_in_destructor();
}

void PipelinesManager::register_defs(std::vector<std::shared_ptr<pipeline::PipelineDefinition>> pipeline_defs)
{
    // Now loop over all and register with the control plane
    for (const auto& pipeline : pipeline_defs)
    {
        protos::PipelineRegisterConfigRequest request;
        protos::PipelineAddMappingRequest mapping_request;

        auto* config  = request.mutable_config();
        auto* mapping = mapping_request.mutable_mapping();

        for (const auto& [segment_id, segment] : pipeline->segments())
        {
            protos::PipelineConfiguration_SegmentConfiguration seg_config;

            seg_config.set_name(segment->name());

            for (const auto& [egress_port_name, egress_port_info] : segment->egress_port_infos())
            {
                // Add it to the list of ports
                (*seg_config.mutable_egress_ports()->Add()) = egress_port_name;

                // See if this manifold has been created already
                if (!config->manifolds().contains(egress_port_name))
                {
                    // Add the manifold
                    protos::PipelineConfiguration_ManifoldConfiguration manifold;

                    manifold.set_name(egress_port_name);
                    manifold.set_type_id(egress_port_info->type_index.hash_code());
                    manifold.set_type_string(type_name(egress_port_info->type_index));

                    config->mutable_manifolds()->emplace(egress_port_name, std::move(manifold));
                }
            }

            for (const auto& [ingress_port_name, ingress_port_info] : segment->ingress_port_infos())
            {
                // Add it to the list of ports
                (*seg_config.mutable_ingress_ports()->Add()) = ingress_port_name;

                // See if this manifold has been created already
                if (!config->manifolds().contains(ingress_port_name))
                {
                    // Add the manifold
                    protos::PipelineConfiguration_ManifoldConfiguration manifold;

                    manifold.set_name(ingress_port_name);
                    manifold.set_type_id(ingress_port_info->type_index.hash_code());
                    manifold.set_type_string(type_name(ingress_port_info->type_index));

                    config->mutable_manifolds()->emplace(ingress_port_name, std::move(manifold));
                }
            }

            config->mutable_segments()->emplace(segment->name(), std::move(seg_config));

            protos::PipelineMapping_SegmentMapping seg_mapping;

            seg_mapping.set_segment_name(segment->name());

            seg_mapping.mutable_by_policy()->set_value(::mrc::protos::SegmentMappingPolicies::OnePerWorker);

            mapping->mutable_segments()->emplace(segment->name(), std::move(seg_mapping));
        }

        auto response = this->runtime().control_plane().await_unary<protos::PipelineRegisterConfigResponse>(
            protos::EventType::ClientUnaryPipelineRegisterConfig,
            request);

        m_definitions[response->pipeline_definition_id()] = pipeline;

        mapping_request.set_definition_id(response->pipeline_definition_id());

        // Now add a mapping to create some pipeline instances
        this->runtime().control_plane().await_unary<protos::PipelineAddMappingResponse>(
            protos::EventType::ClientUnaryPipelineAddMapping,
            mapping_request);
    }
}

pipeline::PipelineDefinition& PipelinesManager::get_definition(uint64_t definition_id)
{
    CHECK(m_definitions.contains(definition_id))
        << "Pipeline with ID: " << definition_id << " not found in registered pipeline definitions";

    return *m_definitions[definition_id];
}

pipeline::PipelineInstance& PipelinesManager::get_instance(uint64_t instance_id)
{
    CHECK(m_instances.contains(instance_id))
        << "Pipeline with ID: " << instance_id << " not found in pipeline instances";

    return *m_instances[instance_id];
}

void PipelinesManager::do_service_start(std::stop_token stop_token)
{
    Promise<void> completed_promise;

    std::stop_callback stop_callback(stop_token, [&completed_promise]() {
        completed_promise.set_value();
    });

    this->mark_started();

    completed_promise.get_future().get();
}

void PipelinesManager::sync_state(const control_plane::state::Connection& connection)
{
    // Before creating/removing, sync the state of all children

    // Check for assignments
    auto cur_pipelines = extract_keys(m_instances);
    auto new_pipelines = extract_keys(connection.assigned_pipelines());

    auto [create_pipelines, remove_pipelines] = compare_difference(cur_pipelines, new_pipelines);

    // construct new segments and attach to manifold
    for (const auto& id : create_pipelines)
    {
        this->create_pipeline(connection.assigned_pipelines().at(id));
    }

    for (const auto& id : remove_pipelines)
    {
        this->erase_pipeline(id);
    }
}

// void PipelinesManager::do_service_start(std::stop_token stop_token)
// {
//     Promise<void> completed_promise;

//     // Now, subscribe to the control plane state updates and filter only on updates to this instance ID
//     m_system_runtime.control_plane()
//         .state_update_obs()
//         .tap([](const control_plane::state::ControlPlaneState& state) {
//             VLOG(10) << "State Update: PipelinesManager";
//         })
//         .subscribe(
//             [this](control_plane::state::ControlPlaneState state) {
//                 // Handle updates to the worker
//                 this->process_state_update(state);
//             },
//             [this](std::exception_ptr ex_ptr) {
//                 try
//                 {
//                     std::rethrow_exception(ex_ptr);
//                 } catch (const std::exception& ex)
//                 {
//                     LOG(ERROR) << "Error in " << this->debug_prefix() << ex.what();
//                 }
//             },
//             [&completed_promise] {
//                 completed_promise.set_value();
//             });

//     // Need to mark this started before waiting
//     this->mark_started();

//     // Yield until the observable is finished
//     completed_promise.get_future().get();
// }

void PipelinesManager::create_pipeline(const control_plane::state::PipelineInstance& instance)
{
    // Get the definition for this instance
    auto def_id = instance.definition().id();

    CHECK(m_definitions.contains(def_id)) << "Unknown definition ID: " << def_id;

    auto definition = m_definitions[def_id];

    // // First, double check if this still needs to be created by trying to activate it
    // auto request = protos::ResourceUpdateStatusRequest();

    // request.set_resource_type("PipelineInstances");
    // request.set_resource_id(pipe_instance_id);
    // request.set_status(protos::ResourceActualStatus::Actual_Creating);

    // auto response = this->runtime().control_plane().await_unary<protos::ResourceUpdateStatusResponse>(
    //     protos::EventType::ClientUnaryResourceUpdateStatus,
    //     request);

    // Resource was activated, lets now create the object
    auto [added_iterator, did_add] = m_instances.emplace(
        instance.id(),
        std::make_shared<pipeline::PipelineInstance>(*this, definition, instance.id()));

    // Now start as a child service
    this->child_service_start(added_iterator->second);
}

void PipelinesManager::erase_pipeline(InstanceID pipeline_id)
{
    CHECK(m_instances.contains(pipeline_id)) << "Invalid state: pipeline does not exist. ID: " << pipeline_id;

    auto& pipeline = m_instances.at(pipeline_id);

    // Stop the pipeline
    pipeline->service_stop();

    // Wait a small time for it to shutdown gracefully
    if (!pipeline->service_await_join(std::chrono::milliseconds(100)))
    {
        LOG(WARNING) << "PipelineInstance[" << pipeline << "] did not stop gracefully. Killing service";

        // Didnt stop gracefully, kill
        pipeline->service_kill();

        // Try waiting one more time, just in case
        pipeline->service_await_join(std::chrono::milliseconds(100));
    }

    CHECK_EQ(m_instances.erase(pipeline_id), 1) << "Invalid state: pipeline not found by ID";
}
}  // namespace mrc::runtime

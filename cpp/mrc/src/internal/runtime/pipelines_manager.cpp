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

#include "internal/pipeline/pipeline.hpp"
#include "internal/segment/definition.hpp"

#include "mrc/core/addresses.hpp"

namespace mrc::internal::runtime {

PipelinesManager::PipelinesManager(control_plane::Client& control_plane_client) :
  m_control_plane_client(control_plane_client)
{}

PipelinesManager::~PipelinesManager() = default;

void PipelinesManager::register_defs(std::vector<std::shared_ptr<pipeline::Pipeline>> pipeline_defs)
{
    // Now loop over all and register with the control plane
    for (const auto& pipeline : pipeline_defs)
    {
        auto request = protos::PipelineRequestAssignmentRequest();

        for (const auto& [segment_id, segment] : pipeline->segments())
        {
            auto* next_seg = request.mutable_segments()->Add();

            next_seg->set_name(segment->name());

            for (const auto& egress_port_name : segment->egress_port_names())
            {
                auto* egress = next_seg->mutable_egress_ports()->Add();
                egress->set_name(egress_port_name);
            }

            for (const auto& ingress_port_name : segment->ingress_port_names())
            {
                auto* ingress = next_seg->mutable_ingress_ports()->Add();
                ingress->set_name(ingress_port_name);
            }

            auto address = segment_address_encode(segment_id, 0);  // rank 0
        }

        // Leave assignments blank for now to allow auto assignment

        auto response = m_control_plane_client.await_unary<protos::PipelineRequestAssignmentResponse>(
            protos::EventType::ClientUnaryRequestPipelineAssignment,
            request);

        m_pipeline_defs[response->pipeline_definition_id()] = pipeline;
    }
}

pipeline::Pipeline& PipelinesManager::get_def(int pipeline_id)
{
    CHECK(m_pipeline_defs.contains(pipeline_id))
        << "Pipeline with ID: " << pipeline_id << " not found in registered pipeline definitions";

    return *m_pipeline_defs[pipeline_id];
}

std::shared_ptr<pipeline::PipelineInstance> PipelinesManager::get_instance(uint64_t definition_id)
{
    // TODO(MDD): Get or create a pipeline instance on demand

    return nullptr;
}

}  // namespace mrc::internal::runtime

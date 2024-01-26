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

#include "internal/pipeline/manifold_instance.hpp"

#include "internal/control_plane/client.hpp"
#include "internal/pipeline/manifold_definition.hpp"
#include "internal/runtime/data_plane_manager.hpp"
#include "internal/runtime/runtime_provider.hpp"

#include "mrc/edge/edge_builder.hpp"
#include "mrc/edge/edge_writable.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/manifold/interface.hpp"
#include "mrc/node/queue.hpp"
#include "mrc/protos/architect.pb.h"
#include "mrc/segment/egress_port.hpp"
#include "mrc/segment/ingress_port.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/string_utils.hpp"

#include <glog/logging.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mrc::runtime {
class Descriptor;
}  // namespace mrc::runtime

namespace mrc::pipeline {

ManifoldInstance::ManifoldInstance(runtime::IInternalRuntimeProvider& runtime,
                                   std::shared_ptr<const ManifoldDefinition> definition,
                                   InstanceID instance_id) :
  SystemResourceManager(runtime, instance_id, (MRC_CONCAT_STR("Manifold[" << definition->name() << "]"))),
  m_definition(std::move(definition))
{}

ManifoldInstance::~ManifoldInstance()
{
    SystemResourceManager::call_in_destructor();
}

const std::string& ManifoldInstance::port_name() const
{
    return m_definition->name();
}

void ManifoldInstance::register_local_output(SegmentAddress address,
                                             std::shared_ptr<segment::IngressPortBase> ingress_port)
{
    // CHECK(!m_local_output.contains(address)) << "Local segment with address: " << address << ", already registered";

    // m_local_output[address] = ingress_port;

    auto incoming_channel = this->runtime().data_plane().get_readable_ingress_channel(address);

    // Save the channel to keep it alive
    m_input_port_nodes[address] = incoming_channel;

    mrc::make_edge(*incoming_channel, ingress_port->get_upstream_sink());
}

void ManifoldInstance::register_local_input(SegmentAddress address,
                                            std::shared_ptr<segment::EgressPortBase> egress_port)
{
    // CHECK(!m_local_input.contains(address)) << "Local segment with address: " << address << ", already registered";
    // egress_port.m_local_input[address] = egress_port;

    auto& outgoing_channel = m_interface->get_input_sink();

    // Probably want to save the channel here

    mrc::make_edge(egress_port->get_downstream_source(), outgoing_channel);
}

void ManifoldInstance::unregister_local_output(SegmentAddress address)
{
    throw std::runtime_error("Not implemented");
}

void ManifoldInstance::unregister_local_input(SegmentAddress address)
{
    throw std::runtime_error("Not implemented");
}

std::shared_ptr<manifold::Interface> ManifoldInstance::get_interface() const
{
    CHECK(m_interface) << "Must start ManifoldInstance before using the interface";

    return m_interface;
}

// void ManifoldInstance::do_service_start(std::stop_token stop_token)
// {
//     m_interface = m_definition->build(this->runnable());

//     m_interface->start();

//     this->mark_started();

//     m_interface->join();
// }

control_plane::state::ManifoldInstance ManifoldInstance::filter_resource(
    const control_plane::state::ControlPlaneState& state) const
{
    if (!state.manifold_instances().contains(this->id()))
    {
        throw exceptions::MrcRuntimeError(MRC_CONCAT_STR("Could not find Manifold Instance with ID: " << this->id()));
    }
    return state.manifold_instances().at(this->id());
}

bool ManifoldInstance::on_created_requested(control_plane::state::ManifoldInstance& instance, bool needs_local_update)
{
    if (needs_local_update)
    {
        m_interface = m_definition->build(this->runnable());
    }

    return true;
}

void ManifoldInstance::on_completed_requested(control_plane::state::ManifoldInstance& instance)
{
    CHECK(m_interface) << "Must create ManifoldInstance before starting";

    m_interface->start();
}

void ManifoldInstance::on_running_state_updated(control_plane::state::ManifoldInstance& instance)
{
    if (m_actual_input_segments == instance.requested_input_segments() &&
        m_actual_output_segments == instance.requested_output_segments())
    {
        // No changes
        return;
    }

    std::vector<manifold::ManifoldPolicyInputInfo> manifold_inputs;
    std::map<SegmentAddress, manifold::ManifoldPolicyOutputInfo> manifold_outputs;

    std::map<InstanceID, std::shared_ptr<edge::IReadableProvider<std::unique_ptr<runtime::ValueDescriptor>>>>
        input_port_nodes;
    std::map<InstanceID, std::shared_ptr<edge::IWritableProvider<std::unique_ptr<runtime::ValueDescriptor>>>>
        output_port_nodes;

    // First, loop over all requested inputs and hook these up so they are available
    for (const auto& [seg_id, is_local] : instance.requested_input_segments())
    {
        if (is_local)
        {
            // Use nullptr if its local (because we never see the internal connection)
            manifold_inputs.emplace_back(seg_id, true, 1, nullptr);
        }
        else
        {
            auto remote_edge = this->runtime().data_plane().get_readable_ingress_channel(seg_id);

            input_port_nodes[seg_id] = remote_edge;

            manifold_inputs.emplace_back(seg_id, false, 1, nullptr);
        }
    }

    // Now loop over the requested outputs
    for (const auto& [seg_id, is_local] : instance.requested_output_segments())
    {
        if (is_local)
        {
            // Gets the same queue that the data plane uses to send data to the manifold
            auto remote_edge = this->runtime().data_plane().get_writable_ingress_channel(seg_id);

            output_port_nodes[seg_id] = remote_edge;

            manifold_outputs.emplace(seg_id, manifold::ManifoldPolicyOutputInfo(seg_id, true, 1, remote_edge.get()));
        }
        else
        {
            // Get an edge from the data plane for this particular, remote segment
            auto remote_edge = this->runtime().data_plane().get_writable_egress_channel(seg_id);

            output_port_nodes[seg_id] = remote_edge;

            manifold_outputs.emplace(seg_id, manifold::ManifoldPolicyOutputInfo(seg_id, false, 1, remote_edge.get()));
        }
    }

    // Update the policy on the manifold interface
    m_interface->update_policy(manifold::ManifoldPolicy(std::move(manifold_inputs), std::move(manifold_outputs)));

    // Now persist the port nodes so they stay alive
    m_input_port_nodes  = std::move(input_port_nodes);
    m_output_port_nodes = std::move(output_port_nodes);

    // Now save the actual values for next iteration
    m_actual_input_segments  = instance.requested_input_segments();
    m_actual_output_segments = instance.requested_output_segments();

    // Send the message that we have updated the actual segments
    auto request = protos::ManifoldUpdateActualAssignmentsRequest();

    request.set_manifold_instance_id(this->id());
    request.mutable_actual_input_segments()->insert(m_actual_input_segments.begin(), m_actual_input_segments.end());
    request.mutable_actual_output_segments()->insert(m_actual_output_segments.begin(), m_actual_output_segments.end());

    auto response =
        this->runtime().control_plane().template await_unary<protos::ManifoldUpdateActualAssignmentsResponse>(
            protos::EventType::ClientUnaryManifoldUpdateActualAssignments,
            request);
}

void ManifoldInstance::add_input(SegmentAddress address, bool is_local)
{
    if (is_local)
    {
        // CHECK(m_local_input.contains(address)) << "Missing local ingress for address: " << address;

        m_local_input[address]->connect_to_manifold(m_interface);
    }
    else
    {
        throw std::runtime_error("Not implemented: add_ingress(remote)");
    }
}

void ManifoldInstance::add_output(SegmentAddress address, bool is_local)
{
    if (is_local)
    {
        // CHECK(m_local_output.contains(address)) << "Missing local egress for address: " << address;

        m_local_output[address]->connect_to_manifold(m_interface);
    }
    else
    {
        throw std::runtime_error("Not implemented: add_egress(remote)");
    }
}

void ManifoldInstance::remove_input(SegmentAddress address)
{
    throw std::runtime_error("Not implemented: remove_ingress");
}

void ManifoldInstance::remove_output(SegmentAddress address)
{
    throw std::runtime_error("Not implemented: remove_egress");
}

}  // namespace mrc::pipeline

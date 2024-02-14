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

#pragma once

#include "internal/control_plane/state/root_state.hpp"
#include "internal/runtime/resource_manager_base.hpp"

#include "mrc/types.hpp"

#include <rxcpp/rx.hpp>

#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <string>

namespace mrc::edge {
template <typename T>
class IWritableProvider;
}  // namespace mrc::edge
namespace mrc::node {
template <typename T>
class Queue;
}  // namespace mrc::node
namespace mrc::runtime {
class Descriptor;
class IInternalRuntimeProvider;
}  // namespace mrc::runtime
namespace mrc::manifold {
struct Interface;
}  // namespace mrc::manifold

namespace mrc::segment {
class EgressPortBase;
class IngressPortBase;
}  // namespace mrc::segment

namespace mrc::manifold {
struct ManifoldAction;
}

namespace mrc::pipeline {
class ManifoldDefinition;

class ManifoldInstance final : public runtime::SystemResourceManager<control_plane::state::ManifoldInstance>
{
  public:
    ManifoldInstance(runtime::IInternalRuntimeProvider& runtime,
                     std::shared_ptr<const ManifoldDefinition> definition,
                     InstanceID instance_id);
    ~ManifoldInstance() override;

    const std::string& port_name() const;

    void register_local_output(PortAddress2 port_address, std::shared_ptr<segment::IngressPortBase> ingress_port);
    void register_local_input(PortAddress2 port_address, std::shared_ptr<segment::EgressPortBase> egress_port);

    void unregister_local_output(PortAddress2 port_address);
    void unregister_local_input(PortAddress2 port_address);

    std::shared_ptr<manifold::Interface> get_interface() const;

  private:
    control_plane::state::ManifoldInstance filter_resource(
        const control_plane::state::ControlPlaneState& state) const override;

    bool on_created_requested(control_plane::state::ManifoldInstance& instance, bool needs_local_update) override;

    void on_completed_requested(control_plane::state::ManifoldInstance& instance) override;

    void on_running_state_updated(control_plane::state::ManifoldInstance& instance) override;

    void on_stopped_requested(control_plane::state::ManifoldInstance& instance) override;

    void add_input(SegmentAddress address, bool is_local);
    void add_output(SegmentAddress address, bool is_local);

    void remove_input(SegmentAddress address);
    void remove_output(SegmentAddress address);

    std::shared_ptr<const ManifoldDefinition> m_definition;

    uint64_t m_instance_id;

    std::shared_ptr<manifold::Interface> m_interface;
    std::shared_ptr<node::WritableEntrypoint<manifold::ManifoldAction>> m_manifold_action_entry;

    std::map<SegmentAddress, std::shared_ptr<segment::IngressPortBase>> m_local_output;
    std::map<SegmentAddress, std::shared_ptr<segment::EgressPortBase>> m_local_input;

    std::map<PortAddress2, std::shared_ptr<edge::IReadableProvider<std::unique_ptr<runtime::ValueDescriptor>>>>
        m_local_input_channels;
    std::map<PortAddress2, std::shared_ptr<edge::IWritableProvider<std::unique_ptr<runtime::ValueDescriptor>>>>
        m_local_output_channels;

    std::map<PortAddress2, std::shared_ptr<edge::IReadableProvider<std::unique_ptr<runtime::ValueDescriptor>>>>
        m_policy_input_channels;
    std::map<PortAddress2, std::shared_ptr<edge::IWritableProvider<std::unique_ptr<runtime::ValueDescriptor>>>>
        m_policy_output_channels;

    std::map<SegmentAddressCombined2, bool> m_actual_input_segments;
    std::map<SegmentAddressCombined2, bool> m_actual_output_segments;
};

}  // namespace mrc::pipeline

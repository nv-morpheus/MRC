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
#include "internal/runtime/runtime_provider.hpp"

#include "mrc/core/async_service.hpp"
#include "mrc/types.hpp"

#include <cstdint>
#include <map>
#include <memory>

namespace mrc::runtime {
class Runtime;
}
namespace mrc::manifold {
struct Interface;
}  // namespace mrc::manifold

namespace mrc::segment {
class EgressPortBase;
class IngressPortBase;
}  // namespace mrc::segment

namespace mrc::pipeline {
class ManifoldDefinition;

class ManifoldInstance final : public runtime::ResourceManagerBase<control_plane::state::ManifoldInstance>
{
  public:
    ManifoldInstance(runtime::IInternalRuntimeProvider& runtime,
                     std::shared_ptr<const ManifoldDefinition> definition,
                     InstanceID instance_id);
    ~ManifoldInstance() override;

    void register_local_output(SegmentAddress address, std::shared_ptr<segment::IngressPortBase> ingress_port);
    void register_local_input(SegmentAddress address, std::shared_ptr<segment::EgressPortBase> egress_port);

    void unregister_local_output(SegmentAddress address);
    void unregister_local_input(SegmentAddress address);

    std::shared_ptr<manifold::Interface> get_interface() const;

  private:
    control_plane::state::ManifoldInstance filter_resource(
        const control_plane::state::ControlPlaneState& state) const override;

    bool on_created_requested(control_plane::state::ManifoldInstance& instance, bool needs_local_update) override;

    void on_completed_requested(control_plane::state::ManifoldInstance& instance) override;

    void on_running_state_updated(control_plane::state::ManifoldInstance& instance) override;

    void add_input(SegmentAddress address, bool is_local);
    void add_output(SegmentAddress address, bool is_local);

    void remove_input(SegmentAddress address);
    void remove_output(SegmentAddress address);

    std::shared_ptr<const ManifoldDefinition> m_definition;

    uint64_t m_instance_id;

    std::shared_ptr<manifold::Interface> m_interface;

    std::map<SegmentAddress, std::shared_ptr<segment::IngressPortBase>> m_local_output;
    std::map<SegmentAddress, std::shared_ptr<segment::EgressPortBase>> m_local_input;

    std::map<SegmentAddress, bool> m_actual_input_segments;
    std::map<SegmentAddress, bool> m_actual_output_segments;
};

}  // namespace mrc::pipeline

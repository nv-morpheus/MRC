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

#include "mrc/runnable/runner.hpp"
#include "mrc/types.hpp"

#include <rxcpp/rx.hpp>

#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace mrc::runnable {
class Launcher;
}  // namespace mrc::runnable
namespace mrc::runtime {
class IInternalPartitionRuntimeProvider;
}  // namespace mrc::runtime

namespace mrc::manifold {
struct Interface;
}  // namespace mrc::manifold

namespace mrc::segment {
class SegmentDefinition;
class BuilderDefinition;

// todo(ryan) - inherit from service
class SegmentInstance final : public runtime::PartitionResourceManager<control_plane::state::SegmentInstance>
{
    using base_t = runtime::PartitionResourceManager<control_plane::state::SegmentInstance>;

  public:
    SegmentInstance(runtime::IInternalPartitionRuntimeProvider& runtime,
                    std::shared_ptr<const SegmentDefinition> definition,
                    SegmentAddress2 segment_address);
    ~SegmentInstance() override;

    SegmentID2 id() const;
    const std::string& name() const;
    SegmentAddress2 address() const;

    PipelineID2 pipeline_id() const;
    // SegmentRank rank() const;

    std::shared_ptr<manifold::Interface> create_manifold(const PortName& name);
    void attach_manifold(std::shared_ptr<manifold::Interface> manifold);

  protected:
    const std::string& info() const;

  private:
    control_plane::state::SegmentInstance filter_resource(
        const control_plane::state::ControlPlaneState& state) const override;

    bool on_created_requested(control_plane::state::SegmentInstance& instance, bool needs_local_update) override;

    void on_completed_requested(control_plane::state::SegmentInstance& instance) override;

    void on_stopped_requested(control_plane::state::SegmentInstance& instance) override;

    void service_start_impl();
    // void do_service_await_live() final;
    // void do_service_stop() final;
    // void do_service_kill() final;
    // void do_service_await_join() final;

    void callback_on_state_change(const std::string& name, const mrc::runnable::Runner::State& new_state);

    // bool set_local_status(control_plane::state::ResourceActualStatus status);

    std::shared_ptr<const SegmentDefinition> m_definition;
    // uint64_t m_pipeline_instance_id;

    // SegmentRank m_rank;
    SegmentAddress2 m_address;

    std::string m_info;

    std::unique_ptr<BuilderDefinition> m_builder;

    control_plane::state::ResourceActualStatus m_local_status{control_plane::state::ResourceActualStatus::Unknown};

    std::map<std::string, std::unique_ptr<mrc::runnable::Launcher>> m_launchers;
    std::map<std::string, std::unique_ptr<mrc::runnable::Runner>> m_runners;
    int64_t m_running_count{0};
    // std::map<std::string, std::unique_ptr<mrc::runnable::Runner>> m_egress_runners;
    // std::map<std::string, std::unique_ptr<mrc::runnable::Runner>> m_ingress_runners;

    mutable std::mutex m_mutex;
};

}  // namespace mrc::segment

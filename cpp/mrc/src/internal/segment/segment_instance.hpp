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
#include "internal/remote_descriptor/manager.hpp"
#include "internal/service.hpp"

#include "mrc/core/async_service.hpp"
#include "mrc/runnable/runner.hpp"
#include "mrc/types.hpp"

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace mrc::runtime {
class PartitionRuntime;
}

namespace mrc::pipeline {
class PipelineResources;
}  // namespace mrc::pipeline
namespace mrc::manifold {
struct Interface;
}  // namespace mrc::manifold

namespace mrc::segment {
class SegmentDefinition;
class BuilderDefinition;

// todo(ryan) - inherit from service
class SegmentInstance final : public AsyncService, public runnable::RunnableResourcesProvider
{
  public:
    SegmentInstance(runtime::PartitionRuntime& runtime,
                    std::shared_ptr<const SegmentDefinition> definition,
                    SegmentAddress instance_id,
                    uint64_t pipeline_instance_id);
    ~SegmentInstance() override;

    const std::string& name() const;
    SegmentID id() const;
    SegmentRank rank() const;
    SegmentAddress address() const;

    std::shared_ptr<manifold::Interface> create_manifold(const PortName& name);
    void attach_manifold(std::shared_ptr<manifold::Interface> manifold);

  protected:
    const std::string& info() const;

  private:
    void do_service_start(std::stop_token stop_token) final;
    void process_state_update(control_plane::state::SegmentInstance& instance);

    void service_start_impl();
    // void do_service_await_live() final;
    // void do_service_stop() final;
    // void do_service_kill() final;
    // void do_service_await_join() final;

    void callback_on_state_change(const std::string& name, const mrc::runnable::Runner::State& new_state);

    std::shared_ptr<const SegmentDefinition> m_definition;
    SegmentAddress m_instance_id;
    uint64_t m_pipeline_instance_id;

    SegmentRank m_rank;
    SegmentAddress m_address;
    std::string m_info;

    std::unique_ptr<BuilderDefinition> m_builder;
    runtime::PartitionRuntime& m_runtime;

    control_plane::state::ResourceStatus m_local_status{control_plane::state::ResourceStatus::Registered};

    std::map<std::string, std::unique_ptr<mrc::runnable::Runner>> m_runners;
    // std::map<std::string, std::unique_ptr<mrc::runnable::Runner>> m_egress_runners;
    // std::map<std::string, std::unique_ptr<mrc::runnable::Runner>> m_ingress_runners;

    mutable std::mutex m_mutex;
};

}  // namespace mrc::segment

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

#pragma once

#include "internal/runtime/runtime_provider.hpp"

#include "mrc/core/async_service.hpp"
#include "mrc/types.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <stop_token>
#include <vector>

namespace mrc::control_plane::state {
struct Connection;
struct PipelineInstance;
}  // namespace mrc::control_plane::state
namespace mrc::pipeline {
class PipelineDefinition;
class PipelineInstance;
}  // namespace mrc::pipeline

namespace mrc::runtime {

/**
 * @brief Partition Resources define the set of Resources available to a given Partition
 *
 * This class does not own the actual resources, that honor is bestowed on the resources::Manager. This class is
 * constructed and owned by the resources::Manager to ensure validity of the references.
 */
class PipelinesManager : public AsyncService, public InternalRuntimeProvider
{
  public:
    PipelinesManager(IInternalRuntimeProvider& runtime);
    ~PipelinesManager() override;

    void register_defs(std::vector<std::shared_ptr<pipeline::PipelineDefinition>> pipeline_defs);

    pipeline::PipelineDefinition& get_definition(uint64_t definition_id);

    pipeline::PipelineInstance& get_instance(uint64_t instance_id);

  private:
    void do_service_start(std::stop_token stop_token) override;

    void sync_state(const control_plane::state::Connection& connection);

    void create_pipeline(const control_plane::state::PipelineInstance& instance);
    void erase_pipeline(InstanceID pipeline_id);

    std::map<uint64_t, std::shared_ptr<pipeline::PipelineDefinition>> m_definitions;

    std::map<uint64_t, std::shared_ptr<pipeline::PipelineInstance>> m_instances;

    friend class ConnectionManager;
};

}  // namespace mrc::runtime

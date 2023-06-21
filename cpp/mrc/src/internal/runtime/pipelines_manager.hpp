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

#include "internal/control_plane/client.hpp"
#include "internal/control_plane/state/root_state.hpp"
#include "internal/pipeline/pipeline_definition.hpp"
#include "internal/pipeline/pipeline_instance.hpp"
#include "internal/resources/partition_resources.hpp"

#include "mrc/core/async_service.hpp"
#include "mrc/types.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace mrc::runtime {

/**
 * @brief Partition Resources define the set of Resources available to a given Partition
 *
 * This class does not own the actual resources, that honor is bestowed on the resources::Manager. This class is
 * constructed and owned by the resources::Manager to ensure validity of the references.
 */
class PipelinesManager : public ResourceManagerBase<control_plane::state::Connection>
{
  public:
    PipelinesManager(Runtime& runtime, InstanceID connection_id);
    ~PipelinesManager() override;

    void register_defs(std::vector<std::shared_ptr<pipeline::PipelineDefinition>> pipeline_defs);

    pipeline::PipelineDefinition& get_definition(uint64_t definition_id);

    pipeline::PipelineInstance& get_instance(uint64_t instance_id);

  private:
    control_plane::state::Connection filter_resource(
        const control_plane::state::ControlPlaneState& state) const override;

    void on_running_state_updated(control_plane::state::Connection& instance) override;

    void create_pipeline(const control_plane::state::PipelineInstance& instance);
    void erase_pipeline(InstanceID pipeline_id);

    std::map<uint64_t, std::shared_ptr<pipeline::PipelineDefinition>> m_definitions;

    std::map<uint64_t, std::unique_ptr<pipeline::PipelineInstance>> m_instances;
};

}  // namespace mrc::runtime

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

#include "internal/async_service.hpp"
#include "internal/control_plane/client.hpp"
#include "internal/pipeline/pipeline.hpp"
#include "internal/pipeline/pipeline_instance.hpp"
#include "internal/resources/partition_resources.hpp"

#include "mrc/types.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace mrc::internal::runtime {

/**
 * @brief Partition Resources define the set of Resources available to a given Partition
 *
 * This class does not own the actual resources, that honor is bestowed on the resources::Manager. This class is
 * constructed and owned by the resources::Manager to ensure validity of the references.
 */
class PipelinesManager : public AsyncService, public runnable::RunnableResourcesProvider
{
  public:
    PipelinesManager(Runtime& system_runtime);
    ~PipelinesManager() override;

    void register_defs(std::vector<std::shared_ptr<pipeline::Pipeline>> pipeline_defs);

    pipeline::Pipeline& get_definition(uint64_t definition_id);

    pipeline::PipelineInstance& get_instance(uint64_t instance_id);

  private:
    void do_service_start(std::stop_token stop_token) final;

    void process_state_update(control_plane::state::ControlPlaneState& state);

    Runtime& m_system_runtime;

    std::map<uint64_t, std::shared_ptr<pipeline::Pipeline>> m_definitions;

    std::map<uint64_t, std::unique_ptr<pipeline::PipelineInstance>> m_instances;
};

}  // namespace mrc::internal::runtime

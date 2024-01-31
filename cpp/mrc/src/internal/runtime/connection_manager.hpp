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

#include "internal/control_plane/state/root_state.hpp"
#include "internal/runtime/resource_manager_base.hpp"

#include "mrc/types.hpp"

#include <rxcpp/rx.hpp>

#include <cstdint>
#include <exception>
#include <map>
#include <memory>

namespace mrc::runtime {

class DataPlaneManager;
class PipelinesManager;
class Runtime;
class WorkerManager;

/**
 * @brief Partition Resources define the set of Resources available to a given Partition
 *
 * This class does not own the actual resources, that honor is bestowed on the resources::Manager. This class is
 * constructed and owned by the resources::Manager to ensure validity of the references.
 */
class ConnectionManager : public SystemResourceManager<control_plane::state::Executor>
{
  public:
    ConnectionManager(Runtime& runtime, InstanceID instance_id);
    ~ConnectionManager() override;

    PipelinesManager& pipelines_manager() const;

  private:
    control_plane::state::Executor filter_resource(const control_plane::state::ControlPlaneState& state) const override;

    bool on_created_requested(control_plane::state::Executor& instance, bool needs_local_update) override;

    void on_completed_requested(control_plane::state::Executor& instance) override;

    void on_running_state_updated(control_plane::state::Executor& instance) override;

    void on_stopped_requested(control_plane::state::Executor& instance) override;

    Runtime& m_runtime;

    std::shared_ptr<DataPlaneManager> m_data_plane_manager;
    std::shared_ptr<PipelinesManager> m_pipelines_manager;
    std::map<uint64_t, std::shared_ptr<WorkerManager>> m_worker_instances;
};

}  // namespace mrc::runtime

/*
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
#include "internal/control_plane/server.hpp"
#include "internal/runtime/partition_manager.hpp"
#include "internal/runtime/partition_runtime.hpp"
#include "internal/runtime/pipelines_manager.hpp"
#include "internal/system/resources.hpp"

#include "mrc/runtime/api.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace mrc::internal::resources {
class Manager;
}  // namespace mrc::internal::resources

namespace mrc::internal::runtime {

/**
 * @brief Implements the public Runtime interface and owns any high-level runtime resources, e.g. the remote descriptor
 * manager which are built on partition resources. The Runtime object is responsible for bringing up and tearing down
 * core resources manager.
 */
class Runtime final : public mrc::runtime::IRuntime, public Service, public system::SystemProvider
{
  public:
    Runtime(const system::SystemProvider& system);

    // Runtime(std::unique_ptr<resources::Manager> resources);
    ~Runtime() override;

    // IRuntime - total number of partitions
    std::size_t partition_count() const final;

    // IRuntime - total number of gpus / gpu partitions
    std::size_t gpu_count() const final;

    // access the partition specific resources for a given partition_id
    PartitionRuntime& partition(std::size_t partition_id) final;

    // access the full set of internal resources
    resources::Manager& resources() const;

    control_plane::Client& control_plane() const;

    PipelinesManager& pipelines_manager() const
    {
        return *m_pipelines_manager;
    }

    void register_pipelines_defs(std::map<int, std::shared_ptr<pipeline::Pipeline>> pipeline_defs);

  private:
    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;

    std::unique_ptr<resources::Manager> m_resources;
    std::vector<std::unique_ptr<PartitionRuntime>> m_partitions;

    std::unique_ptr<control_plane::Server> m_control_plane_server;
    std::unique_ptr<control_plane::Client> m_control_plane_client;

    std::vector<std::unique_ptr<PartitionManager>> m_partition_managers;

    std::unique_ptr<PipelinesManager> m_pipelines_manager;

    // std::map<int, std::shared_ptr<pipeline::Pipeline>> m_registered_pipeline_defs;
};

}  // namespace mrc::internal::runtime

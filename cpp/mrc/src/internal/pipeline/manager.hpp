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

#include "internal/pipeline/pipeline_definition.hpp"
#include "internal/pipeline/types.hpp"
#include "internal/runtime/runtime.hpp"
#include "internal/service.hpp"

#include "mrc/node/writable_entrypoint.hpp"

#include <rxcpp/rx.hpp>

#include <cstdint>
#include <memory>

// IWYU pragma: no_forward_declare mrc::node::WritableEntrypoint

namespace mrc::resources {
class SystemResources;
}  // namespace mrc::resources
namespace mrc::runnable {
class Runner;
}  // namespace mrc::runnable

namespace mrc::pipeline {
class PipelineDefinition;

/**
 * @brief Responsible for coordinating and controlling a Pipeline running on a set of resources/partitions.
 *
 * Given a pipeline definition from the user and a set of system resources partitioned according to user defined
 * options, the Manager object is responsible for constructing PartitionControllers for each partition in the set of
 * resources and optionally wiring up the control plane and data plane for multi-machine pipelines.
 */
class PipelineManager : public Service
{
  public:
    PipelineManager(runtime::Runtime& runtime, std::shared_ptr<PipelineDefinition> pipeline, uint64_t instance_id);
    ~PipelineManager() override;

    const PipelineDefinition& pipeline() const;

    void push_updates(SegmentAddresses&& segment_addresses);

  private:
    void do_service_start() final;
    void do_service_await_live() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_join() final;

    runtime::Runtime& m_runtime;

    std::shared_ptr<PipelineDefinition> m_pipeline;
    uint64_t m_instance_id;
    std::unique_ptr<node::WritableEntrypoint<ControlMessage>> m_update_channel;
    std::unique_ptr<mrc::runnable::Runner> m_controller;
    rxcpp::composite_subscription m_state_subscription;
};

}  // namespace mrc::pipeline

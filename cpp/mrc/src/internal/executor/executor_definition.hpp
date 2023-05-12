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
#include "internal/runtime/runtime.hpp"
#include "internal/service.hpp"
#include "internal/system/system_provider.hpp"

#include "mrc/node/generic_sink.hpp"
#include "mrc/pipeline/executor.hpp"
#include "mrc/protos/architect_state.pb.h"
#include "mrc/types.hpp"

#include <memory>

namespace mrc {
class Options;
}  // namespace mrc
namespace mrc::pipeline {
class IPipeline;
class PipelineManager;
}  // namespace mrc::pipeline
namespace mrc::resources {
class SystemResources;
}  // namespace mrc::resources
namespace mrc::system {
class ThreadingResources;
}  // namespace mrc::system

namespace mrc::executor {

/**
 * @brief Common Executor code used by both the Standalone and Architect Executors
 *
 * Issues #149 will begin to separate some of the functionality of ExeuctorBase into individual components.
 */
class ExecutorDefinition : public pipeline::IExecutor, public Service, public system::SystemProvider
{
  public:
    ExecutorDefinition(std::shared_ptr<Options> options);
    // Executor(std::unique_ptr<system::Resources> resources);
    ~ExecutorDefinition() override;

    static std::shared_ptr<ExecutorDefinition> unwrap(std::shared_ptr<IExecutor> object);

    void register_pipeline(std::shared_ptr<pipeline::IPipeline> pipeline) override;
    void start() override;
    void stop() override;
    void join() override;

  private:
    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;

    std::unique_ptr<runtime::Runtime> m_runtime;
    // std::unique_ptr<resources::Manager> m_resources_manager;
    std::vector<std::shared_ptr<pipeline::PipelineManager>> m_pipeline_managers;

    std::vector<std::shared_ptr<pipeline::PipelineDefinition>> m_registered_pipeline_defs;

    std::unique_ptr<node::LambdaSinkComponent<const protos::ControlPlaneState>> m_update_sink;

    Mutex m_pipelines_mutex;
    CondV m_pipelines_cv;
};

// std::unique_ptr<Executor> make_executor(std::shared_ptr<Options> options);

// std::unique_ptr<Executor> make_executor(std::unique_ptr<system::ThreadingResources> resources);

}  // namespace mrc::executor
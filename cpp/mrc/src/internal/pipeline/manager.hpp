/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/pipeline/pipeline.hpp"
#include "internal/pipeline/types.hpp"
#include "internal/resources/manager.hpp"
#include "internal/service.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/runnable/runner.hpp"

#include <memory>

namespace mrc::internal::pipeline {

/**
 * @brief Responsible for coordinating and controlling a Pipeline running on a set of resources/partitions.
 *
 * Given a pipeline definition from the user and a set of system resources partitioned according to user defined
 * options, the Manager object is responsible for constructing PartitionControllers for each partition in the set of
 * resources and optionally wiring up the control plane and data plane for multi-machine pipelines.
 */
class Manager : public Service
{
  public:
    Manager(std::shared_ptr<Pipeline> pipeline, resources::Manager& resources);
    ~Manager() override;

    const Pipeline& pipeline() const;

    void push_updates(SegmentAddresses&& segment_addresses);

  protected:
    resources::Manager& resources();

  private:
    void do_service_start() final;
    void do_service_await_live() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_join() final;

    resources::Manager& m_resources;
    std::shared_ptr<Pipeline> m_pipeline;
    std::unique_ptr<node::SourceChannelWriteable<ControlMessage>> m_update_channel;
    std::unique_ptr<mrc::runnable::Runner> m_controller;
};

}  // namespace mrc::internal::pipeline

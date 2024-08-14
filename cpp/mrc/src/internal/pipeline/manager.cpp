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

#include "internal/pipeline/manager.hpp"

#include "internal/pipeline/controller.hpp"
#include "internal/pipeline/pipeline_instance.hpp"
#include "internal/pipeline/types.hpp"
#include "internal/resources/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/runnable/runnable_resources.hpp"

#include "mrc/edge/edge_builder.hpp"
#include "mrc/node/writable_entrypoint.hpp"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runnable/runner.hpp"

#include <glog/logging.h>

#include <exception>
#include <memory>
#include <ostream>
#include <utility>

namespace mrc::pipeline {

Manager::Manager(std::shared_ptr<PipelineDefinition> pipeline,
                 resources::Manager& resources,
                 std::function<void(State)> state_change_cb) :
  Service("pipeline::Manager"),
  m_pipeline(std::move(pipeline)),
  m_resources(resources),
  m_state_change_cb(std::move(state_change_cb))
{
    CHECK(m_pipeline);
    CHECK_GE(m_resources.partition_count(), 1);
    service_start();
}

Manager::~Manager()
{
    Service::call_in_destructor();
}

void Manager::change_stage(State new_state)
{
    DVLOG(1) << "Pipeline::Manager - Changing state to " << static_cast<int>(new_state);
    m_state = new_state;
    if (m_state_change_cb)
    {
        m_state_change_cb(m_state);
    }
}

void Manager::push_updates(SegmentAddresses&& segment_addresses)
{
    CHECK(m_update_channel);

    m_update_channel->await_write({ControlMessageType::Update, std::move(segment_addresses)});
}

void Manager::do_service_start()
{
    mrc::runnable::LaunchOptions main;
    main.engine_factory_name = "main";
    main.pe_count            = 1;
    main.engines_per_pe      = 1;

    auto instance    = std::make_unique<PipelineInstance>(m_pipeline, m_resources);
    auto controller  = std::make_unique<Controller>(std::move(instance));
    m_update_channel = std::make_unique<node::WritableEntrypoint<ControlMessage>>();

    // form edge
    mrc::make_edge(*m_update_channel, *controller);

    // launch controller
    auto launcher = resources().partition(0).runnable().launch_control().prepare_launcher(main, std::move(controller));

    // explicit capture and rethrow the error
    launcher->apply([this](mrc::runnable::Runner& runner) {
        runner.on_completion_callback([this](bool ok) {
            if (!ok)
            {
                LOG(ERROR) << "error detected on controller";
            }
        });
    });
    m_controller = launcher->ignition();
    change_stage(State::Run);
}

void Manager::do_service_await_live()
{
    DVLOG(1) << "Pipeline::Manager - await_live";
    m_controller->await_live();
}

void Manager::do_service_stop()
{
    VLOG(10) << "stop: closing update channels";
    change_stage(State::Stop);
    m_update_channel->await_write({ControlMessageType::Stop});
}

void Manager::do_service_kill()
{
    VLOG(10) << "kill: closing update channels; issuing kill to controllers";
    change_stage(State::Stop);
    m_update_channel->await_write({ControlMessageType::Kill});
}

void Manager::do_service_await_join()
{
    change_stage(State::Joined);
    std::exception_ptr ptr;
    try
    {
        m_controller->runnable_as<Controller>().await_on_pipeline();
    } catch (...)
    {
        ptr = std::current_exception();
    }
    m_update_channel.reset();
    m_controller->await_join();
    if (ptr)
    {
        std::rethrow_exception(ptr);
    }
}

resources::Manager& Manager::resources()
{
    return m_resources;
}

const PipelineDefinition& Manager::pipeline() const
{
    CHECK(m_pipeline);
    return *m_pipeline;
}
}  // namespace mrc::pipeline

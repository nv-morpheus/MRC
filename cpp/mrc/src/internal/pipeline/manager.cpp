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

#include "internal/control_plane/client.hpp"
#include "internal/control_plane/resources.hpp"
#include "internal/pipeline/controller.hpp"
#include "internal/pipeline/instance.hpp"
#include "internal/pipeline/types.hpp"
#include "internal/resources/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/utils/contains.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/edge/edge_builder.hpp"
#include "mrc/node/writable_entrypoint.hpp"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runnable/runner.hpp"

#include <glog/logging.h>

#include <exception>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace mrc::internal::pipeline {

PipelineManager::PipelineManager(std::shared_ptr<Pipeline> pipeline,
                                 resources::Manager& resources,
                                 uint64_t instance_id) :
  m_pipeline(std::move(pipeline)),
  m_resources(resources),
  m_instance_id(instance_id)
{
    CHECK(m_pipeline);
    CHECK_GE(m_resources.partition_count(), 1);
    // service_start();
}

PipelineManager::~PipelineManager()
{
    Service::call_in_destructor();
}

void PipelineManager::push_updates(SegmentAddresses&& segment_addresses)
{
    CHECK(m_update_channel);

    m_update_channel->await_write({ControlMessageType::Update, std::move(segment_addresses)});
}

void PipelineManager::do_service_start()
{
    mrc::runnable::LaunchOptions main;
    main.engine_factory_name = "main";
    main.pe_count            = 1;
    main.engines_per_pe      = 1;

    auto instance    = std::make_unique<Instance>(m_pipeline, m_resources);
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

    // // Now subscribe to the state updates
    // m_state_subscription =
    //     m_resources.control_plane()
    //         .client()
    //         .state_update_obs()
    //         .filter([this](const protos::ControlPlaneState& state) {
    //             return std::find(state.pipeline_instances().ids().begin(),
    //                              state.pipeline_instances().ids().end(),
    //                              m_instance_id) != state.pipeline_instances().ids().end();
    //         })
    //         //    .map([this](const protos::ControlPlaneState& state) {
    //         //        return state.pipeline_instances().entities().at(m_instance_id);
    //         //    })
    //         .subscribe([this](const protos::ControlPlaneState& state) {
    //             auto pipeline_instance = state.pipeline_instances().entities().at(m_instance_id);

    //             pipeline::SegmentAddresses segment_addresses;

    //             for (const auto& seg_instance_id : pipeline_instance.segment_ids())
    //             {
    //                 // Get the segment instance object
    //                 auto segment_instance = state.segment_instances().entities().at(seg_instance_id);

    //                 auto address               = segment_address_encode(segment_instance.definition_id(), 0);  //
    //                 rank 0 segment_addresses[address] = 0;  // partition 0;
    //             }

    //             m_update_channel->await_write({ControlMessageType::Update, std::move(segment_addresses)});
    //         });
}

void PipelineManager::do_service_await_live()
{
    m_controller->await_live();
}

void PipelineManager::do_service_stop()
{
    VLOG(10) << "stop: closing update channels";
    m_update_channel->await_write({ControlMessageType::Stop});
}

void PipelineManager::do_service_kill()
{
    VLOG(10) << "kill: closing update channels; issuing kill to controllers";
    m_update_channel->await_write({ControlMessageType::Kill});
}

void PipelineManager::do_service_await_join()
{
    std::exception_ptr ptr;
    try
    {
        m_controller->runnable_as<Controller>().await_on_pipeline();
    } catch (...)
    {
        ptr = std::current_exception();
    }
    m_state_subscription.unsubscribe();
    m_update_channel.reset();
    m_controller->await_join();
    if (ptr)
    {
        std::rethrow_exception(ptr);
    }
}

resources::Manager& PipelineManager::resources()
{
    return m_resources;
}

const Pipeline& PipelineManager::pipeline() const
{
    CHECK(m_pipeline);
    return *m_pipeline;
}
}  // namespace mrc::internal::pipeline

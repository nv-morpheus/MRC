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

#include "internal/runtime/resource_manager_base.hpp"

#include "internal/control_plane/state/root_state.hpp"
#include "internal/pipeline/manifold_instance.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/runnable/runnable_resources.hpp"
#include "internal/runtime/partition_runtime.hpp"
#include "internal/runtime/pipelines_manager.hpp"
#include "internal/segment/builder_definition.hpp"
#include "internal/segment/segment_definition.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/core/async_service.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/manifold/interface.hpp"
#include "mrc/protos/architect_state.pb.h"
#include "mrc/runnable/launchable.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runnable/runner.hpp"
#include "mrc/segment/egress_port.hpp"
#include "mrc/segment/ingress_port.hpp"
#include "mrc/segment/utils.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>

#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

namespace mrc::segment {

SegmentInstance::SegmentInstance(runtime::PartitionRuntime& runtime,
                                 std::shared_ptr<const SegmentDefinition> definition,
                                 SegmentAddress instance_id,
                                 uint64_t pipeline_instance_id) :
  AsyncService(MRC_CONCAT_STR("SegmentInstance[" << instance_id << "]")),
  runnable::RunnableResourcesProvider(runtime),
  m_runtime(runtime),
  m_definition(std::move(definition)),
  m_instance_id(instance_id),
  m_pipeline_instance_id(pipeline_instance_id),
  m_address(instance_id),
  m_rank(std::get<1>(segment_address_decode(instance_id))),
  m_info(::mrc::segment::info(instance_id))
{}

SegmentInstance::~SegmentInstance() = default;

SegmentID SegmentInstance::id() const
{
    return m_definition->id();
}

void SegmentInstance::do_service_start(std::stop_token stop_token)
{
    Promise<void> completed_promise;

    std::stop_callback stop_callback(stop_token, [this]() {
        if (m_local_status >= control_plane::state::ResourceActualStatus::Creating &&
            m_local_status < control_plane::state::ResourceActualStatus::Stopping)
        {
            this->set_local_status(control_plane::state::ResourceActualStatus::Stopping);
        }
    });

    // Now, subscribe to the control plane state updates and filter only on updates to this instance ID
    m_runtime.control_plane()
        .state_update_obs()
        .tap([this](const control_plane::state::ControlPlaneState& state) {
            VLOG(10) << "State Update: SegmentInstance[" << m_address << "/" << m_definition->name() << "]";
        })
        .map([this](control_plane::state::ControlPlaneState state) -> control_plane::state::SegmentInstance {
            return state.segment_instances().at(m_address);
        })
        .take_while([](control_plane::state::SegmentInstance& state) {
            // Process events until the worker is indicated to be destroyed
            return state.state().actual_status() < control_plane::state::ResourceActualStatus::Destroyed;
        })
        .subscribe(
            [this](control_plane::state::SegmentInstance state) {
                // Handle updates to the worker
                this->process_state_update(state);
            },
            [this](std::exception_ptr ex_ptr) {
                try
                {
                    std::rethrow_exception(ex_ptr);
                } catch (const std::exception& ex)
                {
                    LOG(ERROR) << "Error in " << this->debug_prefix() << ex.what();
                }
            },
            [&completed_promise] {
                completed_promise.set_value();
            });

    // Set that we are now created
    this->set_local_status(control_plane::state::ResourceActualStatus::Created);

    // Yield until the observable is finished
    completed_promise.get_future().get();
}

void SegmentInstance::process_state_update(control_plane::state::SegmentInstance& instance)
{
    switch (instance.state().requested_status())
    {
    case control_plane::state::ResourceRequestedStatus::Initialized:
    case control_plane::state::ResourceRequestedStatus::Created: {
        if (m_local_status < control_plane::state::ResourceActualStatus::Creating)
        {
            // If were not created, finish any initialization
        }

        break;
    }
    case control_plane::state::ResourceRequestedStatus::Completed: {
        if (m_local_status < control_plane::state::ResourceActualStatus::Running)
        {
            // If we are activated, we need to setup the instance and then inform the control plane we are ready
            this->service_start_impl();

            // Set us as running
            this->set_local_status(control_plane::state::ResourceActualStatus::Running);

            // Indicate we have started
            this->mark_started();
        }

        break;
    }
    case control_plane::state::ResourceRequestedStatus::Stopped: {
        break;
    }
    case control_plane::state::ResourceRequestedStatus::Destroyed: {
        break;
    }
    case control_plane::state::ResourceRequestedStatus::Unknown:
    default: {
        CHECK(false) << "Unknown worker state: " << static_cast<int>(instance.state().requested_status());
    }
    }
}

bool SegmentInstance::set_local_status(control_plane::state::ResourceActualStatus status)
{
    CHECK_GE(status, m_local_status) << "Cannot set status backwards!";

    // If we are advancing the status, send the update
    if (status > m_local_status)
    {
        // Issue a resource state update to the control plane
        auto request = protos::ResourceUpdateStatusRequest();

        request.set_resource_type("SegmentInstances");
        request.set_resource_id(this->m_instance_id);
        request.set_status(static_cast<protos::ResourceActualStatus>(status));

        auto response = m_runtime.control_plane().await_unary<protos::ResourceUpdateStatusResponse>(
            protos::EventType::ClientUnaryResourceUpdateStatus,
            request);

        m_local_status = status;

        return true;
    }

    return false;
}
}  // namespace mrc::segment

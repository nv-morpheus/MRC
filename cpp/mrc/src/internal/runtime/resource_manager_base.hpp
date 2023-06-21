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

#include "internal/control_plane/client.hpp"
#include "internal/control_plane/state/root_state.hpp"
#include "internal/remote_descriptor/manager.hpp"
#include "internal/runtime/partition_runtime.hpp"
#include "internal/runtime/runtime_provider.hpp"
#include "internal/service.hpp"

#include "mrc/core/async_service.hpp"
#include "mrc/protos/architect_state.pb.h"
#include "mrc/runnable/runner.hpp"
#include "mrc/types.hpp"

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace mrc::runtime {

template <typename ResourceT>
class ResourceManagerBase : public AsyncService, public runtime::InternalRuntimeProvider
{
  public:
    ResourceManagerBase(runtime::IInternalRuntimeProvider& runtime, uint64_t id, std::string name) :
      AsyncService(std::move(name)),
      runtime::InternalRuntimeProvider(runtime),
      m_id(id)
    {
        if constexpr (std::is_same_v<ResourceT, control_plane::state::Connection>)
        {
            m_resource_type = "Connections";
        }
        else if constexpr (std::is_same_v<ResourceT, control_plane::state::Worker>)
        {
            m_resource_type = "Workers";
        }
        else if constexpr (std::is_same_v<ResourceT, control_plane::state::PipelineInstance>)
        {
            m_resource_type = "PipelineInstances";
        }
        else if constexpr (std::is_same_v<ResourceT, control_plane::state::ManifoldInstance>)
        {
            m_resource_type = "ManifoldInstances";
        }
        else if constexpr (std::is_same_v<ResourceT, control_plane::state::SegmentInstance>)
        {
            m_resource_type = "SegmentInstances";
        }
        else
        {
            throw std::runtime_error("Unsupported resource type");
        }
    }
    ~ResourceManagerBase() override = default;

    uint64_t id() const
    {
        return m_id;
    }

  protected:
    void mark_completed()
    {
        CHECK_EQ(m_local_status, control_plane::state::ResourceActualStatus::Running) << "Can only mark completed "
                                                                                         "while running. Use "
                                                                                         "mark_errored() if an issue "
                                                                                         "occurred";

        this->set_local_actual_status(control_plane::state::ResourceActualStatus::Completed);
    }

    void mark_errored()
    {
        throw std::runtime_error("Not implemented: mark_errored");
    }

  private:
    virtual ResourceT filter_resource(const control_plane::state::ControlPlaneState& state) const = 0;

    void do_service_start(std::stop_token stop_token) override
    {
        Promise<void> completed_promise;

        std::stop_callback stop_callback(stop_token, [this]() {
            if (m_local_status >= control_plane::state::ResourceActualStatus::Creating &&
                m_local_status < control_plane::state::ResourceActualStatus::Stopping)
            {
                // Indicate to the control plane we would like to stop
                this->set_remote_requested_status(control_plane::state::ResourceRequestedStatus::Stopped);
            }
        });

        // Now, subscribe to the control plane state updates and filter only on updates to this instance ID
        this->runtime()
            .control_plane()
            .state_update_obs()
            .tap([this](const control_plane::state::ControlPlaneState& state) {
                VLOG(10) << "State Update: " << m_resource_type << "[" << m_id << "]";
            })
            .map([this](control_plane::state::ControlPlaneState state) -> ResourceT {
                return this->filter_resource(state);
            })
            .take_while([](ResourceT& state) {
                // Process events until the worker is indicated to be destroyed
                return state.state().actual_status() < control_plane::state::ResourceActualStatus::Destroyed;
            })
            .subscribe(
                [this](ResourceT state) {
                    // Handle updates to the worker
                    this->process_state_update(state);
                },
                [this, &completed_promise](std::exception_ptr ex_ptr) {
                    try
                    {
                        std::rethrow_exception(ex_ptr);
                    } catch (const std::exception& ex)
                    {
                        LOG(ERROR) << this->debug_prefix() << " Error in subscription. Message: " << ex.what();
                    }

                    this->service_kill();

                    // Must call the completed promise
                    completed_promise.set_value();
                },
                [&completed_promise] {
                    completed_promise.set_value();
                });

        // Set that we are now initialized (Also triggers a state update)
        this->set_local_actual_status(control_plane::state::ResourceActualStatus::Initialized);

        // Yield until the observable is finished
        completed_promise.get_future().get();

        // Indicate that we are stopped
        this->set_local_actual_status(control_plane::state::ResourceActualStatus::Destroyed);
    }

    void do_service_kill() override
    {
        // Inform the service we died unexpectedly
        this->set_local_actual_status(control_plane::state::ResourceActualStatus::Destroyed);
    }

    void process_state_update(ResourceT& instance)
    {
        switch (instance.state().requested_status())
        {
        case control_plane::state::ResourceRequestedStatus::Initialized:
        case control_plane::state::ResourceRequestedStatus::Created: {
            if (m_local_status < control_plane::state::ResourceActualStatus::Creating)
            {
                // Set our local status to Creating to prevent reentries
                this->set_local_actual_status(control_plane::state::ResourceActualStatus::Creating, false);
            }

            bool needs_local_update = m_local_status < control_plane::state::ResourceActualStatus::Created;

            // Call the resource created function
            bool should_update_local = this->on_created_requested(instance, needs_local_update);

            if (needs_local_update && should_update_local)
            {
                this->set_local_actual_status(control_plane::state::ResourceActualStatus::Created);
            }

            break;
        }
        case control_plane::state::ResourceRequestedStatus::Completed: {
            if (m_local_status < control_plane::state::ResourceActualStatus::Running)
            {
                this->on_completed_requested(instance);

                // Set us as running
                this->set_local_actual_status(control_plane::state::ResourceActualStatus::Running);

                // Indicate we have started
                this->mark_started();
            }

            // Now handle normal state changes since we are running
            this->on_running_state_updated(instance);

            break;
        }
        case control_plane::state::ResourceRequestedStatus::Stopped: {
            if (m_local_status < control_plane::state::ResourceActualStatus::Stopping)
            {
                // Set our local status to prevent reentries
                this->set_local_actual_status(control_plane::state::ResourceActualStatus::Stopping, false);

                // Should call service_stop
                this->on_stopped_requested(instance);

                this->set_local_actual_status(control_plane::state::ResourceActualStatus::Stopped);
            }
            break;
        }
        case control_plane::state::ResourceRequestedStatus::Destroyed: {
            CHECK(false) << "Not sure if this should get here";
            break;
        }
        case control_plane::state::ResourceRequestedStatus::Unknown:
        default: {
            CHECK(false) << "Unknown worker state: " << static_cast<int>(instance.state().requested_status());
        }
        }
    }

    bool set_remote_requested_status(control_plane::state::ResourceRequestedStatus status)
    {
        throw std::runtime_error("Not implemented");
        // // Issue a resource state update to the control plane
        // auto request = protos::ResourceUpdateStatusRequest();

        // request.set_resource_type(m_resource_type);
        // request.set_resource_id(this->m_id);
        // request.set_status(static_cast<protos::ResourceActualStatus>(status));

        // auto response = m_runtime.control_plane().await_unary<protos::ResourceUpdateStatusResponse>(
        //     protos::EventType::ClientUnaryResourceUpdateStatus,
        //     request);
    }

    bool set_local_actual_status(control_plane::state::ResourceActualStatus status, bool push_update = true)
    {
        CHECK_GE(status, m_local_status) << "Cannot set status backwards!";

        // If we are advancing the status, send the update
        if (status > m_local_status)
        {
            if (push_update)
            {
                // Issue a resource state update to the control plane
                auto request = protos::ResourceUpdateStatusRequest();

                request.set_resource_type(m_resource_type);
                request.set_resource_id(this->m_id);
                request.set_status(static_cast<protos::ResourceActualStatus>(status));

                auto response =
                    this->runtime().control_plane().template await_unary<protos::ResourceUpdateStatusResponse>(
                        protos::EventType::ClientUnaryResourceUpdateStatus,
                        request);
            }

            m_local_status = status;

            return true;
        }

        return false;
    }

    virtual bool on_created_requested(ResourceT& instance, bool needs_local_update)
    {
        return needs_local_update;
    }

    virtual void on_completed_requested(ResourceT& instance) {}

    virtual void on_running_state_updated(ResourceT& instance) {}

    virtual void on_stopped_requested(ResourceT& instance)
    {
        this->service_stop();
    }

    uint64_t m_id;
    std::string m_resource_type;

    control_plane::state::ResourceActualStatus m_local_status{control_plane::state::ResourceActualStatus::Unknown};
};

}  // namespace mrc::runtime

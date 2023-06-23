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

#include "internal/segment/segment_instance.hpp"

#include "internal/control_plane/state/root_state.hpp"
#include "internal/pipeline/manifold_instance.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/runnable/runnable_resources.hpp"
#include "internal/runtime/partition_runtime.hpp"
#include "internal/runtime/pipelines_manager.hpp"
#include "internal/runtime/runtime_provider.hpp"
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

SegmentInstance::SegmentInstance(runtime::IInternalRuntimeProvider& runtime,
                                 std::shared_ptr<const SegmentDefinition> definition,
                                 SegmentAddress instance_id,
                                 uint64_t pipeline_instance_id) :
  ResourceManagerBase(runtime, instance_id, MRC_CONCAT_STR("SegmentInstance[" << instance_id << "]")),
  m_definition(std::move(definition)),
  m_pipeline_instance_id(pipeline_instance_id),
  m_address(instance_id),
  m_rank(std::get<1>(segment_address_decode(instance_id))),
  m_info(::mrc::segment::info(instance_id))
{}

SegmentInstance::~SegmentInstance() = default;

const std::string& SegmentInstance::name() const
{
    return m_definition->name();
}

SegmentRank SegmentInstance::rank() const
{
    return m_rank;
}

SegmentAddress SegmentInstance::address() const
{
    return m_address;
}

void SegmentInstance::service_start_impl()
{
    // We construct the builder resources here since we are on the correct numa node
    m_builder = std::make_unique<BuilderDefinition>(*this, m_definition, m_address);

    m_builder->initialize();

    // Get a reference to the pipeline instance
    auto& pipeline_instance = this->runtime().pipelines_manager().get_instance(m_pipeline_instance_id);

    // prepare launchers from m_builder
    // std::map<std::string, std::unique_ptr<mrc::runnable::Launcher>> egress_launchers;
    // std::map<std::string, std::unique_ptr<mrc::runnable::Launcher>> ingress_launchers;

    auto apply_callback = [this](std::unique_ptr<mrc::runnable::Launcher>& launcher, std::string name) {
        launcher->apply([this, n = std::move(name)](mrc::runnable::Runner& runner) {
            runner.on_completion_callback([this, n](bool ok) {
                if (!ok)
                {
                    DVLOG(10) << info() << ": detected a failure in node " << n << "; issuing service_kill()";
                    service_kill();
                }

                // Now remove the runner
                CHECK_EQ(m_runners.erase(n), 1) << "Erased wrong number of runners";

                if (m_runners.empty())
                {
                    // We are shut down, call stop on the service
                    this->service_stop();
                }
            });
        });
    };

    for (const auto& [name, node] : m_builder->egress_ports())
    {
        DVLOG(10) << info() << " constructing launcher egress port " << name;

        pipeline_instance.get_manifold_instance(name).register_local_input(m_address, node);

        // node->connect_to_manifold(pipeline_instance.get_manifold(name));

        m_launchers[name] = node->prepare_launcher(this->runnable().launch_control());
        apply_callback(m_launchers[name], name);
    }

    for (const auto& [name, node] : m_builder->nodes())
    {
        DVLOG(10) << info() << " constructing launcher for " << name;
        m_launchers[name] = node->prepare_launcher(this->runnable().launch_control());
        apply_callback(m_launchers[name], name);
    }

    for (const auto& [name, node] : m_builder->ingress_ports())
    {
        DVLOG(10) << info() << " constructing launcher ingress port " << name;

        pipeline_instance.get_manifold_instance(name).register_local_output(m_address, node);

        node->connect_to_manifold(pipeline_instance.get_manifold(name));

        m_launchers[name] = node->prepare_launcher(this->runnable().launch_control());
        apply_callback(m_launchers[name], name);
    }

    DVLOG(10) << info() << " issuing start request";

    // for (const auto& [name, launcher] : egress_launchers)
    // {
    //     DVLOG(10) << info() << " launching egress port " << name;
    //     m_egress_runners[name] = launcher->ignition();
    // }

    for (const auto& [name, launcher] : m_launchers)
    {
        DVLOG(10) << info() << " launching node " << name;
        m_runners[name] = launcher->ignition();
    }

    // for (const auto& [name, launcher] : ingress_launchers)
    // {
    //     DVLOG(10) << info() << " launching ingress port " << name;
    //     m_ingress_runners[name] = launcher->ignition();
    // }

    // egress_launchers.clear();
    // launchers.clear();
    // ingress_launchers.clear();

    DVLOG(10) << info() << " start has been initiated; use the is_running future to await on startup";
}

// void SegmentInstance::do_service_stop()
// {
//     DVLOG(10) << info() << " issuing stop request";

//     // we do not issue stop for port since they are nodes and stop has no effect

//     for (const auto& [name, runner] : m_runners)
//     {
//         DVLOG(10) << info() << " issuing stop for node " << name;
//         runner->stop();
//     }

//     DVLOG(10) << info() << " stop has been initiated; use the is_completed future to await on shutdown";
// }

// void SegmentInstance::do_service_kill()
// {
//     DVLOG(10) << info() << " issuing kill request";

//     for (const auto& [name, runner] : m_ingress_runners)
//     {
//         DVLOG(10) << info() << " issuing kill for ingress port " << name;
//         runner->kill();
//     }

//     for (const auto& [name, runner] : m_runners)
//     {
//         DVLOG(10) << info() << " issuing kill for node " << name;
//         runner->kill();
//     }

//     for (const auto& [name, runner] : m_egress_runners)
//     {
//         DVLOG(10) << info() << " issuing kill for egress port " << name;
//         runner->kill();
//     }

//     DVLOG(10) << info() << " kill has been initiated; use the is_completed future to await on shutdown";
// }

// void SegmentInstance::do_service_await_live()
// {
//     DVLOG(10) << info() << " await_live started";
//     for (const auto& [name, runner] : m_ingress_runners)
//     {
//         DVLOG(10) << info() << " awaiting on ingress port " << name;
//         runner->await_live();
//     }
//     for (const auto& [name, runner] : m_runners)
//     {
//         DVLOG(10) << info() << " awaiting on  " << name;
//         runner->await_live();
//     }
//     for (const auto& [name, runner] : m_egress_runners)
//     {
//         DVLOG(10) << info() << " awaiting on egress port " << name;
//         runner->await_live();
//     }
//     DVLOG(10) << info() << " join complete";
// }

// void SegmentInstance::do_service_await_join()
// {
//     DVLOG(10) << info() << " join started";
//     std::exception_ptr first_exception = nullptr;

//     auto check = [&first_exception](mrc::runnable::Runner& runner) {
//         try
//         {
//             runner.await_join();
//         } catch (...)
//         {
//             if (first_exception == nullptr)
//             {
//                 first_exception = std::current_exception();
//             }
//         }
//     };

//     for (const auto& [name, runner] : m_ingress_runners)
//     {
//         DVLOG(10) << info() << " awaiting on ingress port join to " << name;
//         check(*runner);
//     }
//     for (const auto& [name, runner] : m_runners)
//     {
//         DVLOG(10) << info() << " awaiting on join to " << name;
//         check(*runner);
//     }
//     for (const auto& [name, runner] : m_egress_runners)
//     {
//         DVLOG(10) << info() << " awaiting on egress port join to " << name;
//         check(*runner);
//     }
//     DVLOG(10) << info() << " join complete";
//     if (first_exception)
//     {
//         LOG(ERROR) << "segment::Instance - an exception was caught while awaiting on one or more nodes - rethrowing";
//         rethrow_exception(std::move(first_exception));
//     }
// }

void SegmentInstance::attach_manifold(std::shared_ptr<manifold::Interface> manifold)
{
    auto port_name = manifold->port_name();

    {
        auto search = m_builder->egress_ports().find(port_name);
        if (search != m_builder->egress_ports().end())
        {
            DVLOG(10) << info() << " attaching manifold for egress port " << port_name;
            search->second->connect_to_manifold(std::move(manifold));
            return;
        }
    }

    {
        auto search = m_builder->ingress_ports().find(port_name);
        if (search != m_builder->ingress_ports().end())
        {
            DVLOG(10) << info() << " attaching manifold for ingress port " << port_name;
            search->second->connect_to_manifold(std::move(manifold));
            return;
        }
    }

    LOG(ERROR) << "unable to find an ingress or egress port matching the port_name " << port_name;
    throw exceptions::MrcRuntimeError("invalid manifold for segment");
}

const std::string& SegmentInstance::info() const
{
    return m_info;
}

std::shared_ptr<manifold::Interface> SegmentInstance::create_manifold(const PortName& name)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    DVLOG(10) << info() << " attempting to build manifold for port " << name;
    {
        auto search = m_builder->egress_ports().find(name);
        if (search != m_builder->egress_ports().end())
        {
            return search->second->make_manifold(this->runnable());
        }
    }
    {
        auto search = m_builder->ingress_ports().find(name);
        if (search != m_builder->ingress_ports().end())
        {
            return search->second->make_manifold(this->runnable());
        }
    }
    LOG(FATAL) << info() << " unable to match ingress or egress port name";
    return nullptr;
}

control_plane::state::SegmentInstance SegmentInstance::filter_resource(
    const control_plane::state::ControlPlaneState& state) const
{
    if (!state.segment_instances().contains(this->id()))
    {
        throw exceptions::MrcRuntimeError(MRC_CONCAT_STR("Could not find Segment Instance with ID: " << this->id()));
    }
    return state.segment_instances().at(this->id());
}

bool SegmentInstance::on_created_requested(control_plane::state::SegmentInstance& instance, bool needs_local_update)
{
    if (needs_local_update)
    {
        // We construct the builder resources here since we are on the correct numa node
        m_builder = std::make_unique<BuilderDefinition>(*this, m_definition, m_address);

        m_builder->initialize();

        // Get a reference to the pipeline instance
        auto& pipeline_instance = this->runtime().pipelines_manager().get_instance(m_pipeline_instance_id);

        // prepare launchers from m_builder
        // std::map<std::string, std::unique_ptr<mrc::runnable::Launcher>> egress_launchers;
        // std::map<std::string, std::unique_ptr<mrc::runnable::Launcher>> ingress_launchers;

        auto apply_callback = [this](std::unique_ptr<mrc::runnable::Launcher>& launcher, std::string name) {
            launcher->apply([this, n = std::move(name)](mrc::runnable::Runner& runner) {
                runner.on_completion_callback([this, n](bool ok) {
                    if (!ok)
                    {
                        DVLOG(10) << info() << ": detected a failure in node " << n << "; issuing service_kill()";
                        service_kill();
                    }

                    // Now remove the runner
                    CHECK_EQ(m_runners.erase(n), 1) << "Erased wrong number of runners";

                    if (m_runners.empty())
                    {
                        // We are shut down, advance us to complete
                        this->mark_completed();
                    }
                });
            });
        };

        for (const auto& [name, node] : m_builder->egress_ports())
        {
            DVLOG(10) << info() << " constructing launcher egress port " << name;

            pipeline_instance.get_manifold_instance(name).register_local_input(m_address, node);

            // node->connect_to_manifold(pipeline_instance.get_manifold(name));

            // m_launchers[name] = node->prepare_launcher(this->runnable().launch_control());
            // apply_callback(m_launchers[name], name);
        }

        for (const auto& [name, node] : m_builder->nodes())
        {
            DVLOG(10) << info() << " constructing launcher for " << name;
            m_launchers[name] = node->prepare_launcher(this->runnable().launch_control());
            apply_callback(m_launchers[name], name);
        }

        for (const auto& [name, node] : m_builder->ingress_ports())
        {
            DVLOG(10) << info() << " constructing launcher ingress port " << name;

            pipeline_instance.get_manifold_instance(name).register_local_output(m_address, node);

            // node->connect_to_manifold(pipeline_instance.get_manifold(name));

            // m_launchers[name] = node->prepare_launcher(this->runnable().launch_control());
            // apply_callback(m_launchers[name], name);
        }
    }

    return true;
}

void SegmentInstance::on_completed_requested(control_plane::state::SegmentInstance& instance)
{
    DVLOG(10) << info() << " issuing start request";

    // for (const auto& [name, launcher] : egress_launchers)
    // {
    //     DVLOG(10) << info() << " launching egress port " << name;
    //     m_egress_runners[name] = launcher->ignition();
    // }

    for (const auto& [name, launcher] : m_launchers)
    {
        DVLOG(10) << info() << " launching node " << name;
        m_runners[name] = launcher->ignition();
    }

    m_launchers.clear();

    // for (const auto& [name, launcher] : ingress_launchers)
    // {
    //     DVLOG(10) << info() << " launching ingress port " << name;
    //     m_ingress_runners[name] = launcher->ignition();
    // }

    // egress_launchers.clear();
    // launchers.clear();
    // ingress_launchers.clear();

    DVLOG(10) << info() << " start has been initiated; use the is_running future to await on startup";
}

// void SegmentInstance::do_service_start(std::stop_token stop_token)
// {
//     Promise<void> completed_promise;

//     std::stop_callback stop_callback(stop_token, [this]() {
//         if (m_local_status >= control_plane::state::ResourceActualStatus::Creating &&
//             m_local_status < control_plane::state::ResourceActualStatus::Stopping)
//         {
//             this->set_local_status(control_plane::state::ResourceActualStatus::Stopping);
//         }
//     });

//     // Now, subscribe to the control plane state updates and filter only on updates to this instance ID
//     this->runtime()
//         .control_plane()
//         .state_update_obs()
//         .tap([this](const control_plane::state::ControlPlaneState& state) {
//             VLOG(10) << "State Update: SegmentInstance[" << m_address << "/" << m_definition->name() << "]";
//         })
//         .map([this](control_plane::state::ControlPlaneState state) -> control_plane::state::SegmentInstance {
//             return state.segment_instances().at(m_address);
//         })
//         .take_while([](control_plane::state::SegmentInstance& state) {
//             // Process events until the worker is indicated to be destroyed
//             return state.state().actual_status() < control_plane::state::ResourceActualStatus::Destroyed;
//         })
//         .subscribe(
//             [this](control_plane::state::SegmentInstance state) {
//                 // Handle updates to the worker
//                 this->process_state_update(state);
//             },
//             [this](std::exception_ptr ex_ptr) {
//                 try
//                 {
//                     std::rethrow_exception(ex_ptr);
//                 } catch (const std::exception& ex)
//                 {
//                     LOG(ERROR) << "Error in " << this->debug_prefix() << ex.what();
//                 }
//             },
//             [&completed_promise] {
//                 completed_promise.set_value();
//             });

//     // Set that we are now created
//     this->set_local_status(control_plane::state::ResourceActualStatus::Created);

//     // Yield until the observable is finished
//     completed_promise.get_future().get();
// }

// void SegmentInstance::process_state_update(control_plane::state::SegmentInstance& instance)
// {
//     switch (instance.state().requested_status())
//     {
//     case control_plane::state::ResourceRequestedStatus::Initialized:
//     case control_plane::state::ResourceRequestedStatus::Created: {
//         if (m_local_status < control_plane::state::ResourceActualStatus::Creating)
//         {
//             // If were not created, finish any initialization
//         }

//         break;
//     }
//     case control_plane::state::ResourceRequestedStatus::Completed: {
//         if (m_local_status < control_plane::state::ResourceActualStatus::Running)
//         {
//             // If we are activated, we need to setup the instance and then inform the control plane we are ready
//             this->service_start_impl();

//             // Set us as running
//             this->set_local_status(control_plane::state::ResourceActualStatus::Running);

//             // Indicate we have started
//             this->mark_started();
//         }

//         break;
//     }
//     case control_plane::state::ResourceRequestedStatus::Stopped: {
//         break;
//     }
//     case control_plane::state::ResourceRequestedStatus::Destroyed: {
//         break;
//     }
//     case control_plane::state::ResourceRequestedStatus::Unknown:
//     default: {
//         CHECK(false) << "Unknown worker state: " << static_cast<int>(instance.state().requested_status());
//     }
//     }
// }

// bool SegmentInstance::set_local_status(control_plane::state::ResourceActualStatus status)
// {
//     CHECK_GE(status, m_local_status) << "Cannot set status backwards!";

//     // If we are advancing the status, send the update
//     if (status > m_local_status)
//     {
//         // Issue a resource state update to the control plane
//         auto request = protos::ResourceUpdateStatusRequest();

//         request.set_resource_type("SegmentInstances");
//         request.set_resource_id(this->m_instance_id);
//         request.set_status(static_cast<protos::ResourceActualStatus>(status));

//         auto response = this->runtime().control_plane().await_unary<protos::ResourceUpdateStatusResponse>(
//             protos::EventType::ClientUnaryResourceUpdateStatus,
//             request);

//         m_local_status = status;

//         return true;
//     }

//     return false;
// }
}  // namespace mrc::segment

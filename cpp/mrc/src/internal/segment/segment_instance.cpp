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
#include "internal/pipeline/resources.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/runtime/partition_runtime.hpp"
#include "internal/segment/builder.hpp"
#include "internal/segment/definition.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/manifold/interface.hpp"
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

namespace mrc::internal::segment {

SegmentInstance::SegmentInstance(runtime::PartitionRuntime& runtime,
                                 std::shared_ptr<const Definition> definition,
                                 SegmentAddress instance_id) :
  AsyncService(MRC_CONCAT_STR("SegmentInstance[" << instance_id << "]")),
  runnable::RunnableResourcesProvider(runtime),
  m_runtime(runtime),
  m_name(definition->name()),
  m_id(definition->id()),
  m_rank(std::get<1>(segment_address_decode(instance_id))),
  m_address(instance_id),
  m_info(::mrc::segment::info(instance_id))
{
    // construct the segment definition on the intended numa node
    m_builder = m_runtime.resources()
                    .runnable()
                    .main()
                    .enqueue([&]() mutable {
                        return std::make_unique<Builder>(m_runtime, definition, m_address);
                    })
                    .get();
}

SegmentInstance::~SegmentInstance() = default;

const std::string& SegmentInstance::name() const
{
    return m_name;
}

const SegmentID& SegmentInstance::id() const
{
    return m_id;
}

const SegmentRank& SegmentInstance::rank() const
{
    return m_rank;
}

const SegmentAddress& SegmentInstance::address() const
{
    return m_address;
}

void SegmentInstance::service_start_impl()
{
    // prepare launchers from m_builder
    std::map<std::string, std::unique_ptr<mrc::runnable::Launcher>> launchers;
    std::map<std::string, std::unique_ptr<mrc::runnable::Launcher>> egress_launchers;
    std::map<std::string, std::unique_ptr<mrc::runnable::Launcher>> ingress_launchers;

    auto apply_callback = [this](std::unique_ptr<mrc::runnable::Launcher>& launcher, std::string name) {
        launcher->apply([this, n = std::move(name)](mrc::runnable::Runner& runner) {
            runner.on_completion_callback([this, n](bool ok) {
                if (!ok)
                {
                    DVLOG(10) << info() << ": detected a failure in node " << n << "; issuing service_kill()";
                    service_kill();
                }
            });
        });
    };

    for (const auto& [name, node] : m_builder->nodes())
    {
        DVLOG(10) << info() << " constructing launcher for " << name;
        launchers[name] = node->prepare_launcher(m_runtime.resources().runnable().launch_control());
        apply_callback(launchers[name], name);
    }

    for (const auto& [name, node] : m_builder->egress_ports())
    {
        DVLOG(10) << info() << " constructing launcher egress port " << name;
        egress_launchers[name] = node->prepare_launcher(m_runtime.resources().runnable().launch_control());
        apply_callback(egress_launchers[name], name);
    }

    for (const auto& [name, node] : m_builder->ingress_ports())
    {
        DVLOG(10) << info() << " constructing launcher ingress port " << name;
        ingress_launchers[name] = node->prepare_launcher(m_runtime.resources().runnable().launch_control());
        apply_callback(ingress_launchers[name], name);
    }

    DVLOG(10) << info() << " issuing start request";

    for (const auto& [name, launcher] : egress_launchers)
    {
        DVLOG(10) << info() << " launching egress port " << name;
        m_egress_runners[name] = launcher->ignition();
    }

    for (const auto& [name, launcher] : launchers)
    {
        DVLOG(10) << info() << " launching node " << name;
        m_runners[name] = launcher->ignition();
    }

    for (const auto& [name, launcher] : ingress_launchers)
    {
        DVLOG(10) << info() << " launching ingress port " << name;
        m_ingress_runners[name] = launcher->ignition();
    }

    egress_launchers.clear();
    launchers.clear();
    ingress_launchers.clear();

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
            return search->second->make_manifold(m_runtime.resources().runnable());
        }
    }
    {
        auto search = m_builder->ingress_ports().find(name);
        if (search != m_builder->ingress_ports().end())
        {
            return search->second->make_manifold(m_runtime.resources().runnable());
        }
    }
    LOG(FATAL) << info() << " unable to match ingress or egress port name";
    return nullptr;
}

void SegmentInstance::do_service_start(std::stop_token stop_token)
{
    Promise<void> completed_promise;

    // Now, subscribe to the control plane state updates and filter only on updates to this instance ID
    m_runtime.control_plane()
        .state_update_obs()
        .tap([this](const control_plane::state::ControlPlaneState& state) {
            VLOG(10) << "State Update: SegmentInstance[" << m_address << "]";
        })
        .map([this](control_plane::state::ControlPlaneState state) -> control_plane::state::SegmentInstance {
            return state.segment_instances().at(m_address);
        })
        .take_while([](control_plane::state::SegmentInstance& state) {
            // Process events until the worker is indicated to be destroyed
            return state.state().status() < control_plane::state::ResourceStatus::Destroyed;
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
                } catch (std::exception ex)
                {
                    LOG(ERROR) << "Error in " << this->debug_prefix() << ex.what();
                }
            },
            [&completed_promise] {
                completed_promise.set_value();
            });

    // Yield until the observable is finished
    completed_promise.get_future().wait();
}

void SegmentInstance::process_state_update(control_plane::state::SegmentInstance& instance)
{
    if (instance.state().status() == control_plane::state::ResourceStatus::Activated)
    {
        // If we are activated, we need to setup the instance and then inform the control plane we are ready
        this->service_start_impl();

        auto request = protos::ResourceUpdateStatusRequest();

        request.set_resource_type("SegmentInstances");
        request.set_resource_id(instance.id());
        request.set_status(protos::ResourceStatus::Ready);

        auto response = m_runtime.control_plane().await_unary<protos::ResourceUpdateStatusResponse>(
            protos::EventType::ClientUnaryResourceUpdateStatus,
            request);

        CHECK(response->ok()) << "Failed to set PipelineInstance to Ready";
    }
}
}  // namespace mrc::internal::segment
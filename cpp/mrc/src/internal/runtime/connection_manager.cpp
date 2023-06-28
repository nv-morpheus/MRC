/**
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

#include "internal/runtime/connection_manager.hpp"

#include "internal/control_plane/state/root_state.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/runnable/runnable_resources.hpp"
#include "internal/runtime/data_plane_manager.hpp"
#include "internal/runtime/partition_runtime.hpp"
#include "internal/runtime/pipelines_manager.hpp"
#include "internal/runtime/resource_manager_base.hpp"
#include "internal/runtime/runtime.hpp"
#include "internal/runtime/runtime_provider.hpp"
#include "internal/runtime/segments_manager.hpp"
#include "internal/runtime/worker_manager.hpp"
#include "internal/segment/segment_definition.hpp"
#include "internal/system/partition.hpp"
#include "internal/ucx/worker.hpp"
#include "internal/utils/ranges.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/core/async_service.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/protos/architect.pb.h"
#include "mrc/protos/architect_state.pb.h"
#include "mrc/types.hpp"
#include "mrc/utils/string_utils.hpp"

#include <glog/logging.h>
#include <google/protobuf/util/message_differencer.h>
#include <rxcpp/rx.hpp>

#include <chrono>
#include <memory>

namespace mrc::runtime {

ConnectionManager::ConnectionManager(Runtime& runtime, InstanceID instance_id) :
  SystemResourceManager(runtime, instance_id, MRC_CONCAT_STR("ConnectionManager[" << instance_id << "]")),
  m_runtime(runtime)
{}

ConnectionManager::~ConnectionManager()
{
    SystemResourceManager::call_in_destructor();
}

PipelinesManager& ConnectionManager::pipelines_manager() const
{
    return *m_pipelines_manager;
}

control_plane::state::Connection ConnectionManager::filter_resource(
    const control_plane::state::ControlPlaneState& state) const
{
    if (!state.connections().contains(this->id()))
    {
        throw exceptions::MrcRuntimeError(MRC_CONCAT_STR("Could not find Worker with ID: " << this->id()));
    }
    return state.connections().at(this->id());
}

bool ConnectionManager::on_created_requested(control_plane::state::Connection& instance, bool needs_local_update)
{
    if (needs_local_update)
    {
        // Create the pipeline manager first
        m_pipelines_manager = std::make_unique<PipelinesManager>(*this);

        this->child_service_start(*m_pipelines_manager, true);

        // // Send the message to create the workers
        // // First thing, need to register this worker with the control plane
        // protos::RegisterWorkersRequest req;

        // for (size_t i = 0; i < m_runtime.resources().partition_count(); ++i)
        // {
        //     req.add_ucx_worker_addresses(m_runtime.resources().partition(i).ucx()->worker().address());
        // }

        // auto resp = m_runtime.control_plane().await_unary<protos::RegisterWorkersResponse>(
        //     protos::ClientUnaryRegisterWorkers,
        //     std::move(req));

        // CHECK_EQ(resp->instance_ids_size(), m_runtime.partition_count());

        // // Now, preemptively create the workers but do not start them (easier to associate the workers index with
        // their
        // // ID)
        // for (size_t i = 0; i < resp->instance_ids_size(); ++i)
        // {
        //     auto worker_id = resp->instance_ids(0);

        //     auto [added_iterator, did_add] = m_worker_instances.emplace(
        //         worker_id,
        //         std::make_unique<WorkerManager>(m_runtime.partition(i), worker_id));
        // }
    }

    // // Now check the state to see if we need to start any workers
    // auto cur_workers = extract_keys(m_worker_instances);
    // auto new_workers = extract_keys(instance.workers());

    // auto [create_workers, remove_workers] = compare_difference(cur_workers, new_workers);

    // // construct new segments and attach to manifold
    // for (const auto& worker_id : create_workers) {}

    // // detach from manifold or stop old segments
    // for (const auto& address : remove_segments)
    // {
    //     // DVLOG(10) << info() << ": stop segment for address " << ::mrc::segment::info(address);
    //     this->erase_segment(address);
    // }

    return true;
}

void ConnectionManager::on_completed_requested(control_plane::state::Connection& instance)
{
    LOG(INFO) << "Connection on_completed_requested";
    // // Activate our worker
    // protos::RegisterWorkersResponse resp;
    // resp.set_machine_id(0);
    // resp.add_instance_ids(m_worker_id);

    // // Need to activate our worker
    // this->runtime().control_plane().await_unary<protos::Ack>(protos::ClientUnaryActivateStream, std::move(resp));
}

void ConnectionManager::on_running_state_updated(control_plane::state::Connection& instance)
{
    m_pipelines_manager->sync_state(instance);
}

void ConnectionManager::on_stopped_requested(control_plane::state::Connection& instance)
{
    this->service_stop();
}

}  // namespace mrc::runtime

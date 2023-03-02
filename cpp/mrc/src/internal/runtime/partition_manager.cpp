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

#include "internal/runtime/partition_manager.hpp"

#include "rxcpp/subjects/rx-subject.hpp"

#include "internal/async_service.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/system/partition.hpp"
#include "internal/ucx/worker.hpp"

#include "mrc/protos/architect.pb.h"

#include <glog/logging.h>
#include <google/protobuf/util/message_differencer.h>

namespace mrc::internal::runtime {

PartitionManager::PartitionManager(resources::PartitionResources& resources,
                                   control_plane::Client& control_plane_client) :
  AsyncService(resources.runnable()),
  m_resources(resources),
  m_control_plane_client(control_plane_client)
{}

PartitionManager::~PartitionManager() = default;

void PartitionManager::do_service_start(std::stop_token stop_token)
{
    // First thing, need to register this worker with the control plane
    protos::RegisterWorkersRequest req;

    req.add_ucx_worker_addresses(m_resources.ucx()->worker().address());

    auto resp = m_control_plane_client.await_unary<protos::RegisterWorkersResponse>(protos::ClientUnaryRegisterWorkers,
                                                                                    std::move(req));

    CHECK_EQ(resp->instance_ids_size(), 1);

    m_instance_id = resp->instance_ids(0);

    // Now, subscribe to the control plane state updates and filter only on updates to this instance ID
    m_control_plane_client.state_update_obs()
        .map([this](protos::ControlPlaneState state) -> mrc::protos::Worker {
            return state.workers().entities().at(m_instance_id);
        })
        .take_while([stop_token](mrc::protos::Worker& worker) {
            // Process events until the worker is indicated to be destroyed
            return worker.state() < protos::WorkerStates::Destroyed && !stop_token.stop_requested();
        })
        .distinct_until_changed([](const mrc::protos::Worker& curr, const mrc::protos::Worker& prev) {
            return google::protobuf::util::MessageDifferencer::Equals(curr, prev);
        })
        .as_blocking()
        .subscribe([this](mrc::protos::Worker worker) {
            // Handle updates to the worker
            this->process_state_update(worker);
        });

    // Now that we are unsubscribed, drop the worker
    protos::TaggedInstance msg;
    msg.set_instance_id(m_instance_id);

    CHECK(m_control_plane_client.await_unary<protos::Ack>(protos::ClientUnaryDropWorker, std::move(msg)));
}

// void PartitionManager::do_service_stop() {}

// void PartitionManager::do_service_kill() {}

// void PartitionManager::do_service_await_live()
// {
//     m_live_promise.get_future().wait();
// }

// void PartitionManager::do_service_await_join()
// {
//     m_shutdown_future.wait();
// }

void PartitionManager::process_state_update(mrc::protos::Worker& worker)
{
    if (worker.state() == protos::WorkerStates::Registered)
    {
        protos::RegisterWorkersResponse resp;
        resp.set_machine_id(0);
        resp.add_instance_ids(m_instance_id);

        // Need to activate our worker
        m_control_plane_client.await_unary<protos::Ack>(protos::ClientUnaryActivateStream, std::move(resp));

        // Indicate this is now live
        this->mark_started();
    }
    else if (worker.state() == protos::WorkerStates::Activated)
    {
        // Check for assignments
        for (auto seg_id : worker.assigned_segment_ids()) {}
    }
    else if (worker.state() == protos::WorkerStates::Deactivated)
    {
        // Handle deactivation
    }
    else
    {
        CHECK(false) << "Unknown worker state: " << worker.state();
    }
}
}  // namespace mrc::internal::runtime

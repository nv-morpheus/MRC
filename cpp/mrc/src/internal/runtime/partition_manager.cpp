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

#include "internal/async_service.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/system/partition.hpp"
#include "internal/ucx/worker.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/protos/architect.pb.h"
#include "mrc/types.hpp"

#include <glog/logging.h>
#include <google/protobuf/util/message_differencer.h>
#include <rxcpp/rx.hpp>

namespace mrc::internal::runtime {

PartitionManager::PartitionManager(resources::PartitionResources& resources,
                                   control_plane::Client& control_plane_client,
                                   PipelinesManager& pipelines_manager) :
  AsyncService(resources.runnable()),
  m_resources(resources),
  m_control_plane_client(control_plane_client),
  m_pipelines_manager(pipelines_manager),
  m_partition_id(resources.partition_id())
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

    m_worker_id = resp->instance_ids(0);

    Promise<void> completed_promise;

    // Now, subscribe to the control plane state updates and filter only on updates to this instance ID
    m_control_plane_client.state_update_obs()
        .map([this](control_plane::state::ControlPlaneState state) -> control_plane::state::Worker {
            return state.workers().at(m_worker_id);
        })
        .take_while([stop_token](control_plane::state::Worker& worker) {
            // Process events until the worker is indicated to be destroyed
            return worker.state() < control_plane::state::WorkerStates::Destroyed && !stop_token.stop_requested();
        })
        .distinct_until_changed([](const control_plane::state::Worker& curr, const control_plane::state::Worker& prev) {
            return curr == prev;
        })
        .subscribe(
            [this](control_plane::state::Worker worker) {
                // Handle updates to the worker
                this->process_state_update(worker);
            },
            [&completed_promise] {
                completed_promise.set_value();
            });

    // Yield until the observable is finished
    completed_promise.get_future().wait();

    // Now that we are unsubscribed, drop the worker
    protos::TaggedInstance msg;
    msg.set_instance_id(m_worker_id);

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

void PartitionManager::process_state_update(mrc::internal::control_plane::state::Worker& worker)
{
    if (worker.state() == control_plane::state::WorkerStates::Registered)
    {
        protos::RegisterWorkersResponse resp;
        resp.set_machine_id(0);
        resp.add_instance_ids(m_worker_id);

        // Need to activate our worker
        m_control_plane_client.await_unary<protos::Ack>(protos::ClientUnaryActivateStream, std::move(resp));

        // Indicate this is now live
        this->mark_started();
    }
    else if (worker.state() == control_plane::state::WorkerStates::Activated)
    {
        // Check for assignments
        auto cur_segments = extract_keys(m_segments);
        auto new_segments = extract_keys(worker.assigned_segments());

        // set of segments to remove
        std::set<SegmentAddress> create_segments;
        std::set_difference(new_segments.begin(),
                            new_segments.end(),
                            cur_segments.begin(),
                            cur_segments.end(),
                            std::inserter(create_segments, create_segments.end()));
        DVLOG(10) << create_segments.size() << " segments will be created";

        // set of segments to remove
        std::set<SegmentAddress> remove_segments;
        std::set_difference(cur_segments.begin(),
                            cur_segments.end(),
                            new_segments.begin(),
                            new_segments.end(),
                            std::inserter(remove_segments, remove_segments.end()));
        DVLOG(10) << remove_segments.size() << " segments marked for removal";

        // construct new segments and attach to manifold
        for (const auto& address : create_segments)
        {
            // auto partition_id = new_segments_map.at(address);
            // DVLOG(10) << info() << ": create segment for address " << ::mrc::segment::info(address)
            //           << " on resource partition: " << partition_id;
            this->create_segment(worker.assigned_segments().at(address).pipeline().id(), address);
        }

        // detach from manifold or stop old segments
        for (const auto& address : remove_segments)
        {
            // DVLOG(10) << info() << ": stop segment for address " << ::mrc::segment::info(address);
            this->erase_segment(address);
        }
    }
    else if (worker.state() == control_plane::state::WorkerStates::Deactivated)
    {
        // Handle deactivation
    }
    else
    {
        CHECK(false) << "Unknown worker state: " << static_cast<int>(worker.state());
    }
}
void PartitionManager::create_segment(uint64_t pipeline_id, SegmentAddress address)
{
    m_resources.runnable()
        .main()
        .enqueue([this, pipeline_id, address] {
            // Get a reference to the pipeline we are creating the segment in
            auto& pipeline_def = m_pipelines_manager.get_def(pipeline_id);

            auto pipeline_instance = m_pipelines_manager.get_instance(pipeline_id);

            //     auto search = m_segments.find(address);
            // CHECK(search == m_segments.end());

            auto [id, rank] = segment_address_decode(address);
            auto definition = pipeline_def.find_segment(id);
            auto segment    = std::make_unique<segment::Instance>(definition, rank, m_resources, m_partition_id);

            // for (const auto& name : definition->egress_port_names())
            // {
            //     VLOG(10) << ::mrc::segment::info(address) << " configuring manifold for egress port " << name;
            //     std::shared_ptr<manifold::Interface> manifold = get_manifold(name);
            //     if (!manifold)
            //     {
            //         VLOG(10) << ::mrc::segment::info(address) << " creating manifold for egress port " << name;
            //         manifold          = segment->create_manifold(name);
            //         m_manifolds[name] = manifold;
            //     }
            //     segment->attach_manifold(manifold);
            // }

            // for (const auto& name : definition->ingress_port_names())
            // {
            //     VLOG(10) << ::mrc::segment::info(address) << " configuring manifold for ingress port " << name;
            //     std::shared_ptr<manifold::Interface> manifold = get_manifold(name);
            //     if (!manifold)
            //     {
            //         VLOG(10) << ::mrc::segment::info(address) << " creating manifold for ingress port " << name;
            //         manifold          = segment->create_manifold(name);
            //         m_manifolds[name] = manifold;
            //     }
            //     segment->attach_manifold(manifold);
            // }

            // m_segments[address] = std::move(segment);
        })
        .get();
}
}  // namespace mrc::internal::runtime

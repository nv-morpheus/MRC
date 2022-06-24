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

#include <srf/protos/architect.pb.h>
#include <srf/types.hpp>  // for MachineID, InstanceID

#include <nvrpc/context.h>
#include <nvrpc/interfaces.h>            // for Resources
#include <nvrpc/life_cycle_streaming.h>  // StreamingContext is an alias for BaseContext<LifeCycleStreaming>
#include <nvrpc/thread_pool.h>

#include <glog/logging.h>

#include <chrono>  // for milliseconds
#include <condition_variable>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <utility>  // for move
#include <vector>

namespace srf::internal::control_plane {

struct UpdateState
{
    std::vector<protos::Event> start_events;
    std::vector<protos::Event> completed_events;

    bool update_started{false};
    bool update_completed{false};

    bool shutdown{false};
};

struct ServerResources : public nvrpc::Resources
{
    using ServerStream =  // NOLINT
        typename nvrpc::StreamingContext<protos::Event, protos::Event, ServerResources>::ServerStream;

    ServerResources();

    void stop(const MachineID& machine_id);

    bool register_workers(const protos::RegisterWorkersRequest&,
                          protos::RegisterWorkersResponse&,
                          std::shared_ptr<ServerStream>);
    bool register_pipeline(const protos::RegisterPipelineRequest&, protos::RegisterPipelineResponse&);

    bool lookup_workers(const protos::LookupWorkersRequest&, protos::LookupWorkersResponse&) const;

    void on_client_update_start(protos::Event&& event);
    void on_client_update_complete(protos::Event&& event);

    void remove_machine(ServerStream* stream);

    MachineID machine_id_from_stream(ServerStream* stream) const;

    void enqueue_event(protos::Event&&);

    void shutdown();

    bool validate_machine_id(const MachineID&, ServerStream*) const;

    template <typename Response>  // NOLINT
    void issue_response(const protos::Event& request, Response&& response);
    void issue_response(const protos::Event& request, protos::Event&& response = protos::Event());

  protected:
    void evaluate_pipeline_state();

    ServerStream& stream(const MachineID&);

    void remove_machine(const MachineID&);

    bool can_start();

    void oracle();

    bool process_event(const protos::Event&);

  private:
    // provides the instance ids
    InstanceID m_instance_id_counter{42};

    // provides node/machine ids
    MachineID m_machine_counter{0};

    // unique worker addresses
    std::set<std::string> m_ucx_worker_addresses;

    // maps instance_ids [keys] to unique node/machine [value]
    std::map<InstanceID, MachineID> m_instance_ids_to_machine_id;
    std::map<MachineID, std::set<InstanceID>> m_machine_id_to_instance_ids;

    std::map<InstanceID, std::string> m_ucx_worker_address_by_id;

    // streams indexed by machine id
    std::map<MachineID, std::shared_ptr<ServerStream>> m_streams_by_id;
    std::map<MachineID, std::shared_ptr<ServerStream>> m_streams_marked_to_stop;

    // update state
    std::unique_ptr<UpdateState> m_update_state{nullptr};

    // machines marked for removal in this batching window
    std::set<MachineID> m_machines_marked_to_stop;

    std::map<MachineID, protos::UpdateAssignments> m_current_assignments;

    std::string m_graphviz;

    // assignment manager
    // AssignmentManager m_assignment_manager;

    // thread pool for oob evaluator / oracle
    ::nvrpc::ThreadPool m_thread_pool;

    // queue for pushing events that require the evaluator / oracle
    std::queue<protos::Event> m_event_queue;

    // future that will be completed when the orcale finishes
    std::future<void> m_oracle_future;
    std::condition_variable m_oracle_cv;
    std::chrono::milliseconds m_oracle_batching_window;
    bool m_oracle_running{true};

    // primary lock for global state
    mutable std::mutex m_mutex;
};

template <typename Response>  // NOLINT
void ServerResources::issue_response(const protos::Event& request, Response&& response)
{
    protos::Event event;
    CHECK(event.mutable_message()->PackFrom(response));
    issue_response(request, std::move(event));
}

}  // namespace srf::internal::control_plane

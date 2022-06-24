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

// #include "internal/control_plane/server_resources.hpp"

// #include "srf/protos/architect.grpc.pb.h"  // IWYU pragma: keep
// #include "srf/types.hpp"                   // for MachineID, InstanceID

// #include <nvrpc/life_cycle_streaming.h>  // StreamingContext is an alias for BaseContext<LifeCycleStreaming>
// #include <nvrpc/thread_pool.h>

// #include <google/protobuf/any.pb.h>

// #include <algorithm>
// #include <chrono>
// #include <condition_variable>
// #include <ostream>      // needed for logging
// #include <type_traits>  // for usage of remove_reference which appears to be used implicitly by issue_response
// #include <vector>

// namespace srf::internal::control_plane {

// bool ServerResources::validate_machine_id(const MachineID& machine_id, ServerStream* stream_ptr) const
// {
//     auto search = m_streams_by_id.find(machine_id);
//     if (search == m_streams_by_id.end())
//     {
//         return false;
//     }
//     return (search->second.get() == stream_ptr);
// }

// void ServerResources::shutdown()
// {
//     DVLOG(10) << "triggering oracle to shutdown";
//     {
//         std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//         CHECK(m_oracle_running);
//         m_oracle_running = false;
//     }
//     m_oracle_cv.notify_all();
//     m_oracle_future.get();
//     DVLOG(10) << "oracle completed";
// }

// ServerResources::ServerStream& ServerResources::stream(const MachineID& id)
// {
//     auto search = m_streams_by_id.find(id);
//     if (search == m_streams_by_id.end())
//     {
//         auto search_marked = m_streams_marked_to_stop.find(id);
//         CHECK(search_marked != m_streams_marked_to_stop.end());
//         return (*search_marked->second);
//     }
//     return *(search->second);
// }

// void ServerResources::issue_response(const protos::Event& request, protos::Event&& response)
// {
//     response.set_event(protos::EventType::Response);
//     response.set_promise(request.promise());
//     stream(request.machine_id()).WriteResponse(std::move(response));
// }

// void ServerResources::on_client_update_start(protos::Event&& event)
// {
//     std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//     CHECK(m_update_state);
//     CHECK(m_update_state->update_started == false);

//     m_update_state->start_events.push_back(event);

//     if (m_update_state->start_events.size() == m_streams_by_id.size())
//     {
//         m_update_state->update_started = true;

//         // issue unblocking update to all machines
//         for (const auto& req : m_update_state->start_events)
//         {
//             issue_response(req);
//         }

//         m_update_state->start_events.clear();
//     }
// }

// void ServerResources::on_client_update_complete(protos::Event&& event)
// {
//     std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//     CHECK(m_update_state);
//     CHECK(m_update_state->update_started);
//     CHECK(m_update_state->update_completed == false);

//     m_update_state->completed_events.push_back(event);

//     if (m_update_state->completed_events.size() == m_streams_by_id.size())
//     {
//         m_update_state->update_completed = true;

//         // as soon as this is issued, the pipeline are running again
//         // issue unblocking update to all machines
//         for (const auto& req : m_update_state->completed_events)
//         {
//             issue_response(req);
//         }

//         // machines marked as stop will be moved from m_stream_by_id to m_streams_marked_to_stop
//         for (const auto& id : m_machines_marked_to_stop)
//         {
//             VLOG(1) << "machine " << id << " marked as stopped; no future updates will issued";
//             auto stream                  = m_streams_by_id.at(id);
//             m_streams_marked_to_stop[id] = stream;
//             m_streams_by_id.erase(id);
//         }
//         m_machines_marked_to_stop.clear();

//         m_update_state->completed_events.clear();

//         // when the lock is released, a new updated can be started
//         m_update_state.reset();
//         m_oracle_cv.notify_all();
//     }
// }

// ServerResources::ServerResources() : m_thread_pool(1), m_oracle_batching_window(std::chrono::seconds(2))
// {
//     m_oracle_future = m_thread_pool.enqueue([this] { oracle(); });
// }

// void ServerResources::oracle()
// {
//     std::unique_lock<decltype(m_mutex)> lock(m_mutex);

//     // set_current_thread_name("oracle");

//     bool has_event        = false;
//     bool pipeline_started = false;
//     auto deadline         = std::chrono::steady_clock::now() + m_oracle_batching_window;

//     while (m_oracle_running)
//     {
//         has_event = false;
//         m_machines_marked_to_stop.clear();

//         m_oracle_cv.wait(lock, [this] { return !m_oracle_running || !m_event_queue.empty(); });

//         if (!m_oracle_running)
//         {
//             CHECK_EQ(m_event_queue.size(), 0);
//             return;
//         }

//         deadline = std::chrono::steady_clock::now() + m_oracle_batching_window;

//         DVLOG(10) << "oracle batching window starting - processing events";

//         while (std::chrono::steady_clock::now() < deadline)
//         {
//             while (!m_event_queue.empty())
//             {
//                 has_event = true;
//                 process_event(m_event_queue.front());
//                 m_event_queue.pop();
//             }

//             // yield the lock to let more events get queued
//             m_oracle_cv.wait_until(lock, deadline, [this] { return m_event_queue.size(); });
//         }

//         DVLOG(10) << "oracle batching window complete - issuing update";

//         if (has_event && m_oracle_running)
//         {
//             if (pipeline_started || m_assignment_manager.can_start())
//             {
//                 // the first time this executed means the pipeline will start executing after
//                 // the update is complete
//                 pipeline_started = true;

//                 // this evaluates the global state and issue updates to all registered machines
//                 CHECK(m_update_state == nullptr);
//                 evaluate_pipeline_state();

//                 // block the oracle thread from making further progress on the event queue
//                 // until the architect has completed the the current update
//                 CHECK(m_update_state != nullptr);
//                 m_oracle_cv.wait(lock, [this] { return m_update_state == nullptr; });
//             }
//         }
//     }
// }

// bool ServerResources::process_event(const protos::Event& request)
// {
//     // the global state lock is owned when entering this method

//     protos::Event response;

//     switch (request.event())
//     {
//     case protos::EventType::ControlStop:
//         DVLOG(1) << "processing event ControlStop from machine " << request.machine_id();
//         stop(request.machine_id());
//         m_machines_marked_to_stop.insert(request.machine_id());
//         issue_response(request);
//         return true;
//         break;

//     case protos::EventType::ClientSegmentsOnComplete: {
//         DVLOG(1) << "processing event ClientSegmentsOnComplete from machine " << request.machine_id();
//         protos::OnComplete complete_request;
//         CHECK(request.message().UnpackTo(&complete_request));
//         for (const auto& segment_address : complete_request.segment_addresses())
//         {
//             m_assignment_manager.segment_on_complete(segment_address);
//         }
//         issue_response(request, protos::ControlMessage());
//         return true;
//     }
//     break;

//     case protos::EventType::ClientRegisterPipeline: {
//         DVLOG(1) << "processing event ClientRegisterPipeline from machine " << request.machine_id();
//         protos::RegisterPipelineRequest pipeline_request;
//         protos::RegisterPipelineResponse pipeline_response;
//         CHECK(request.message().UnpackTo(&pipeline_request));
//         CHECK(register_pipeline(pipeline_request, pipeline_response));
//         issue_response(request, std::move(pipeline_response));
//         DVLOG(10) << "completed pipeline_registation for machine " << pipeline_request.machine_id();
//         return true;
//     }
//     break;

//     default:
//         LOG(WARNING) << "Unknowned streaming event provided";
//     }

//     return false;
// }

// void ServerResources::enqueue_event(protos::Event&& event)
// {
//     std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//     CHECK(m_oracle_running);
//     m_event_queue.push(std::move(event));
//     m_oracle_cv.notify_one();
// }

// void ServerResources::stop(const MachineID& machine_id)
// {
//     CHECK(m_update_state == nullptr);
//     m_assignment_manager.remove_machine(machine_id);
// }

// MachineID ServerResources::machine_id_from_stream(ServerStream* stream) const
// {
//     for (const auto& stream_kv : m_streams_by_id)
//     {
//         if (stream_kv.second.get() == stream)
//         {
//             return stream_kv.first;
//         }
//     }

//     for (const auto& stream_kv : m_streams_marked_to_stop)
//     {
//         if (stream_kv.second.get() == stream)
//         {
//             return stream_kv.first;
//         }
//     }

//     LOG(FATAL) << "shoudl be unreachable";
//     return 0;
// }

// void ServerResources::remove_machine(ServerStream* stream)
// {
//     auto machine_id = machine_id_from_stream(stream);
//     remove_machine(machine_id);
// }

// void ServerResources::remove_machine(const MachineID& machine_id)
// {
//     std::lock_guard<decltype(m_mutex)> lock(m_mutex);

//     auto instances = m_machine_id_to_instance_ids.find(machine_id);
//     CHECK(instances != m_machine_id_to_instance_ids.end());

//     // pipelines are stored per machine id
//     m_assignment_manager.remove_machine(machine_id);

//     for (const auto& instance_id : instances->second)
//     {
//         auto worker_address = m_ucx_worker_address_by_id.find(instance_id);
//         CHECK(worker_address != m_ucx_worker_address_by_id.end());

//         CHECK_EQ(m_ucx_worker_addresses.erase(worker_address->second), 1);
//         m_ucx_worker_address_by_id.erase(worker_address);
//     }

//     m_machine_id_to_instance_ids.erase(instances);

//     // re-evaluate global state - for now propogate the shutdown message to all
//     // todo(ryan) - update global state - this could be async updates to all other nodes
//     // so this is a method that will be awaited on and will yield the fiber
//     // - x issue updates to machines that are not shutting down
//     // - x wait on global update

//     m_streams_by_id.erase(machine_id);
//     m_streams_marked_to_stop.erase(machine_id);
// }

// bool ServerResources::register_workers(const protos::RegisterWorkersRequest& req,
//                                        protos::RegisterWorkersResponse& resp,
//                                        std::shared_ptr<ServerStream> stream)
// {
//     std::lock_guard<decltype(m_mutex)> lock(m_mutex);

//     CHECK(stream);
//     DVLOG(1) << "registering srf instance with " << req.ucx_worker_addresses_size() << " resource groups";

//     // validate that the worker addresses are valid before updating state
//     for (const auto& worker_address : req.ucx_worker_addresses())
//     {
//         auto search = m_ucx_worker_addresses.find(worker_address);
//         if (search != m_ucx_worker_addresses.end())
//         {
//             LOG(WARNING) << "received a connection request for a ucx worker that is already registered";
//             return false;
//         }
//     }

//     // set machine id for the current stream
//     resp.set_machine_id(m_machine_counter);
//     m_streams_by_id[m_machine_counter] = stream;

//     // assigned and register instance ids fore each ucx worker address
//     for (const auto& worker_address : req.ucx_worker_addresses())
//     {
//         auto id = m_instance_id_counter++;

//         m_ucx_worker_addresses.insert(worker_address);
//         m_ucx_worker_address_by_id[id] = worker_address;

//         m_instance_ids_to_machine_id[id] = m_machine_counter;
//         m_machine_id_to_instance_ids[m_machine_counter].insert(id);

//         DVLOG(5) << "adding instance id to response: " << id << " assigned to unique node: " << m_machine_counter;
//         resp.add_instance_ids(id);
//     }

//     m_machine_counter++;

//     return true;
// }

// bool ServerResources::register_pipeline(const protos::RegisterPipelineRequest& req,
//                                         protos::RegisterPipelineResponse& resp)
// {
//     CHECK(!m_assignment_manager.has_pipeline(req.machine_id()));
//     for (const auto& req_group : req.requested_config())
//     {
//         auto instance_id = req_group.instance_id();
//         CHECK(!m_assignment_manager.has_pipeline_config(instance_id));
//     }

//     m_assignment_manager.add_pipeline(req.machine_id(), req.pipeline());

//     for (const auto& req_group : req.requested_config())
//     {
//         m_assignment_manager.add_pipeline_config(req.machine_id(), req_group);
//     }

//     return true;
// }

// void ServerResources::evaluate_pipeline_state()
// {
//     // CHECK(m_assignment_manager.can_start());
//     CHECK(m_update_state == nullptr);
//     m_update_state = std::make_unique<UpdateState>();

//     protos::UpdateAssignments update_message;
//     auto assignments = m_assignment_manager.evaluate_state();

//     for (const auto& [segment_address, assignment] : assignments)
//     {
//         *update_message.add_assignments() = assignment;
//     }

//     for (const auto& [machine_id, stream] : m_streams_by_id)
//     {
//         protos::Event event;
//         event.set_event(protos::EventType::ServerUpdateAssignments);
//         event.set_machine_id(machine_id);
//         event.mutable_message()->PackFrom(update_message);
//         DVLOG(10) << "issuing update event for machine " << machine_id;
//         stream->WriteResponse(std::move(event));
//     }

//     VLOG(1) << "updates events have been issued for all machines - awaiting responses";
// }

// bool ServerResources::lookup_workers(const protos::LookupWorkersRequest& request,
//                                      protos::LookupWorkersResponse& response) const
// {
//     for (const auto& instance_id : request.instance_ids())
//     {
//         auto search = m_ucx_worker_address_by_id.find(instance_id);
//         if (search == m_ucx_worker_address_by_id.end())
//         {
//             LOG(WARNING) << "client requesting working address for unknown instance id: " << instance_id;
//             continue;
//         }
//         auto* worker_address = response.add_worker_addresses();
//         worker_address->set_instance_id(instance_id);
//         worker_address->set_worker_address(search->second);
//     }

//     return true;
// }

// }  // namespace srf::internal::control_plane

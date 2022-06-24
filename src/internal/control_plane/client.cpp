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

// #include "internal/control_plane/client.hpp"

// #include "internal/base/placement.hpp"
// #include "internal/base/placement_resources.hpp"
// #include "internal/data_plane/instance.hpp"
// #include "internal/data_plane/server.hpp"
// #include "internal/pipeline.hpp"

// #include <srf/protos/architect.grpc.pb.h>
// #include "srf/channel/status.hpp"
// #include "srf/node/generic_sink.hpp"
// #include "srf/node/source_channel.hpp"
// #include "srf/runnable/context.hpp"
// #include "srf/runnable/fiber_context.hpp"
// #include "srf/runnable/forward.hpp"
// #include "srf/runnable/launch_control.hpp"
// #include "srf/runnable/launch_options.hpp"
// #include "srf/runnable/launcher.hpp"
// #include "srf/runnable/runner.hpp"
// #include "srf/types.hpp"

// #include <nvrpc/client/client_fiber_streaming.h>

// #include <boost/fiber/future/promise.hpp>
// #include <boost/fiber/operations.hpp>
// #include <rxcpp/rx-predef.hpp>
// #include <rxcpp/rx-subscriber.hpp>

// #include <chrono>
// #include <functional>
// #include <memory>
// #include <ostream>
// #include <type_traits>

// namespace srf::internal::control_plane {

// using StreamingClient = nvrpc::client::fiber::ClientStreaming<protos::Event, protos::Event>;  // NOLINT

// /**
//  * @brief Specialized FiberRunnable and InternalService to execute the gRPC client's progress engine
//  *
//  * By creating a specialized Runnable that is also an InternalService, this enables users optionally control and
//  * specialize the runtime location of this Runnable via the Options.
//  *
//  */
// class GrpcClientProgressEngine final : public runnable::FiberRunnable<>
// {
//   public:
//     GrpcClientProgressEngine(std::function<void()> progress_engine) : m_progress_engine(std::move(progress_engine))
//     {}

//   private:
//     void run(runnable::FiberContext<>& ctx) final
//     {
//         m_progress_engine();
//     }

//     std::function<void()> m_progress_engine;
// };

// /**
//  * @brief Event handler sink that receives updates from the architect server and applies those update to the local
//  * pipeline instance
//  *
//  */
// class EventHandlerUpdateAssignments final : public node::GenericSink<protos::Event>
// {
//   public:
//     EventHandlerUpdateAssignments(pipeline::Instance& pipeline, Client& client) : m_pipeline(pipeline),
//     m_client(client)
//     {}

//   private:
//     void on_data(protos::Event&& event) final
//     {
//         CHECK(event.event() == protos::EventType::ServerUpdateAssignments)
//             << "unexpected event type received; handler only processes ServerUpdateAssignments events";
//     }

//     pipeline::Instance& m_pipeline;
//     Client& m_client;
// };

// std::unique_ptr<::grpc::ClientAsyncReaderWriter<protos::Event, protos::Event>> Client::PrepareAsync(
//     ::grpc::ClientContext* context, ::grpc::CompletionQueue* cq)
// {
//     CHECK(m_stub);
//     return m_stub->PrepareAsyncEventStream(context, cq);
// }

// void Client::TimeoutBackoff(const std::uint64_t& backoff)
// {
//     if (backoff < 16384)
//     {
//         boost::this_fiber::yield();
//     }
//     else
//     {
//         auto deadline = std::chrono::high_resolution_clock::now() + std::chrono::nanoseconds(backoff);
//         boost::this_fiber::sleep_until(deadline);
//     }
// }

// void Client::CallbackOnInitialized()
// {
//     m_promise_live.set_value();
// }

// void Client::CallbackOnComplete(const ::grpc::Status& status)
// {
//     m_promise_complete.set_value(status);
// }

// void Client::CallbackOnResponseReceived(protos::Event&& event)
// {
//     protos::Event response;

//     switch (event.event())
//     {
//         // handle a subset of events directly on the grpc event loop

//     case protos::EventType::Response: {
//         auto* promise = reinterpret_cast<Promise<protos::Event>*>(event.promise());
//         if (promise != nullptr)
//         {
//             promise->set_value(std::move(event));
//         }
//     }
//     break;

//         // handle all other events on the event sink

//     default:
//         LOG(FATAL) << "event channel not implemented";
//         m_event_channel->await_write(std::move(event));
//     }
// }

// Client::Client(std::shared_ptr<ArchitectRuntime> runtime, std::shared_ptr<protos::Architect::Stub> stub) :
//   m_runtime(std::move(runtime)),
//   m_event_channel(std::make_unique<node::SourceChannelWriteable<protos::Event>>()),
//   m_stub(std::move(stub))
// {
//     // kick off grpc progress engine for streaming client
//     // we specify the `srf_network` fiber engine factory for all network runnables, this is not overridable
//     DVLOG(10) << "[streaming_client: init] preparing to launch progress engine";
//     runnable::LaunchOptions grpc_options;
//     m_grpc_progress_engine =
//         m_runtime->resources(0)
//             .launch_control()
//             .prepare_launcher(runnable::LaunchOptions("srf_network"),
//                               std::make_unique<GrpcClientProgressEngine>([this] { StreamingClient::ProgressEngine();
//                               }))
//             ->ignition();

//     DVLOG(10) << "[streaming_client: init] progress engine: launched";
//     m_grpc_progress_engine->await_live();
//     DVLOG(10) << "[streaming_client: init] progress engine: running";

//     // construct any event handler (sinks) and connect them to their respective event sources/writers
//     // m_event_handler_update_assignments = std::make_unique<EventHandlerUpdateAssignments>(*this);

//     DVLOG(10) << "[streaming_client: init] grpc stream initialize";
//     StreamingClient::Initialize();

//     DVLOG(10) << "[streaming_client: init] awaiting grpc initialization";
//     m_promise_live.get_future().get();
//     DVLOG(10) << "[streaming_client: init] grpc initialized";

//     DVLOG(10) << "[streaming_client: init] register machine/ucx worker addresses with the architect";
//     register_workers();
// }

// Client::~Client()
// {
//     DVLOG(10) << info() << "closing steam";
//     StreamingClient::CloseWrites();
//     DVLOG(10) << info() << "awaiting completion";
//     auto status = m_promise_complete.get_future().get();
//     DVLOG(10) << info() << "steam completed";
//     CHECK(status.ok());

//     DVLOG(10) << info() << "shutting down cq";
//     StreamingClient::Shutdown();
//     DVLOG(10) << info() << "awaiting progress engine join";
//     m_grpc_progress_engine->await_join();
//     DVLOG(10) << info() << "progress engine complete";
// }

// void Client::register_workers()
// {
//     // Register UCX worker addresses with the Architect and receive assigned InstanceIDs
//     // Request:  list of uxc worker addresses
//     // Response: list of instance ids

//     protos::RegisterWorkersRequest request;

//     for (int i = 0; i < m_runtime->placement().group_count(); ++i)
//     {
//         // register ucx event managers worker address - this is incoming ucx events that are handled
//         request.add_ucx_worker_addresses(m_runtime->data_plane_instance(i).events_manager().worker_address());
//     }

//     CHECK_EQ(request.ucx_worker_addresses_size(), m_runtime->placement().group_count());

//     auto response =
//         await_unary<protos::RegisterWorkersResponse>(protos::EventType::ClientRegisterWorkers, std::move(request));

//     m_machine_id = response.machine_id();

//     std::stringstream ss;
//     ss << "[streaming_client: " << m_machine_id << "] ";
//     m_info = ss.str();
//     VLOG(1) << info() << "architect assigned this machine as " << m_machine_id;

//     VLOG(1) << info() << "each numa domaim is assigned a globally unique instance_id";
//     for (const auto& id : response.instance_ids())
//     {
//         VLOG(1) << info() << "instance_id:" << id;
//         m_instance_ids.push_back(id);
//     }
// }

// }  // namespace srf::internal::control_plane

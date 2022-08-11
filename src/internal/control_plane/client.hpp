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

#include "internal/control_plane/client/instance.hpp"
#include "internal/control_plane/client/subscription_service.hpp"
#include "internal/expected.hpp"
#include "internal/grpc/client_streaming.hpp"
#include "internal/grpc/progress_engine.hpp"
#include "internal/grpc/promise_handler.hpp"
#include "internal/grpc/stream_writer.hpp"
#include "internal/resources/forward.hpp"
#include "internal/runnable/engines.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/service.hpp"
#include "internal/ucx/common.hpp"

#include "srf/exceptions/runtime_error.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/operators/broadcast.hpp"
#include "srf/node/operators/router.hpp"
#include "srf/node/source_properties.hpp"
#include "srf/protos/architect.grpc.pb.h"
#include "srf/protos/architect.pb.h"
#include "srf/runnable/runner.hpp"
#include "srf/types.hpp"
#include "srf/utils/macros.hpp"

#include <grpcpp/completion_queue.h>

#include <mutex>

namespace srf::internal::control_plane {

template <typename ResponseT>
class AsyncStatus;

/**
 * @brief Primary Control Plane Client
 *
 * A single instance of Client should be instantiated per processes. This class is responsible owning the client side
 * bidirectional async grpc stream, server event handler, and router used to push server side events to partition client
 * event handlers. This class may also create a grpc::CompletionQueue and run a progress engine and progress handler if
 * constructed without an external CQ being passed in. If a CQ is provided, then it is assumed the progress engine and
 * progress handler are also external.
 *
 * The event handler with this class will directly handle ClientErrors, while InstanceErrors will be forward via the
 * event router to the specific instance handler.
 *
 */
class Client final : public Service
{
  public:
    enum class State
    {
        Disconnected,
        FailedToConnect,
        Connected,
        RegisteringWorkers,
        Operational,
    };

    using stream_t         = std::shared_ptr<rpc::ClientStream<srf::protos::Event, srf::protos::Event>>;
    using writer_t         = std::shared_ptr<rpc::StreamWriter<srf::protos::Event>>;
    using event_t          = stream_t::element_type::IncomingData;
    using update_channel_t = srf::node::SourceChannelWriteable<protos::StateUpdate>;

    Client(runnable::Resources& runnable);

    // if we already have an grpc progress engine running, we don't need run another, just use that cq
    Client(runnable::Resources& runnable, std::shared_ptr<grpc::CompletionQueue> cq);

    ~Client() final;

    const State& state() const
    {
        return m_state;
    }

    MachineID machine_id() const;
    const std::vector<InstanceID>& instance_ids() const;

    std::map<InstanceID, std::unique_ptr<client::Instance>> register_ucx_addresses(
        std::vector<std::optional<ucx::Resources>>& ucx_resources);

    // void register_port_publisher(InstanceID instance_id, const std::string& port_name);
    // void register_port_subscriber(InstanceID instance_id, const std::string& port_name);
    client::SubscriptionService& get_or_create_subscription_service(std::string name, std::set<std::string> roles);

    template <typename ResponseT, typename RequestT>
    Expected<ResponseT> await_unary(const protos::EventType& event_type, RequestT&& request);

    template <typename ResponseT, typename RequestT>
    void async_unary(const protos::EventType& event_type, RequestT&& request, AsyncStatus<ResponseT>& status);

    template <typename MessageT>
    void issue_event(const protos::EventType& event_type, MessageT&& message);

    void issue_event(const protos::EventType& event_type);

    bool has_subscription_service(const std::string& name) const;

    const runnable::LaunchOptions& launch_options() const;

    void drop_instance(const InstanceID& instance_id);

    // request that the server start an update
    void request_update();

    // returns a future which will be completed on the next server update
    // Future<void> await_update();

  private:
    void route_state_update_event(const event_t& event);
    void route_state_update(std::uint64_t tag, protos::StateUpdate&& update);

    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;
    void do_handle_event(event_t&& event);

    void forward_state(State state);

    State m_state{State::Disconnected};

    MachineID m_machine_id;
    std::vector<InstanceID> m_instance_ids;
    std::map<InstanceID, std::unique_ptr<update_channel_t>> m_update_channels;
    // std::map<InstanceID, std::shared_ptr<client::Instance>> m_instances;

    runnable::Resources& m_runnable;

    std::shared_ptr<grpc::CompletionQueue> m_cq;
    std::shared_ptr<grpc::Channel> m_channel;
    std::shared_ptr<srf::protos::Architect::Stub> m_stub;

    // if true, then the following runners should not be null
    // if false, then the following runners must be null
    bool m_owns_progress_engine;
    std::unique_ptr<srf::runnable::Runner> m_progress_handler;
    std::unique_ptr<srf::runnable::Runner> m_progress_engine;
    std::unique_ptr<srf::runnable::Runner> m_event_handler;
    std::map<std::string, std::unique_ptr<client::SubscriptionService>> m_subscription_services;

    // Stream Context
    stream_t m_stream;

    // StreamWriter acquired from m_stream->await_init()
    // The customer destruction of this object will cause a gRPC WritesDone to be issued to the server.
    writer_t m_writer;

    runnable::LaunchOptions m_launch_options;

    // when the server starts an update epoch, it will start by sending a proto::ServerStateUpdateStart event
    // and conclude by sending a protos::ServerStateUpdateFinish
    // after receiving a start and before a finish, this value should be true
    // this value, in conjunction with the awaiting_update, can be used to determine if a request update
    // message should be sent to the server
    // todo(ryan) - move to a separate class/state machine to decouple logic from client
    bool m_update_in_progress{false};
    bool m_update_requested{false};

    mutable std::mutex m_mutex;

    friend network::Resources;
};

// todo: create this object from the client which will own the stop_source
// create this object with a stop_token associated with the client's stop_source

template <typename ResponseT>
class AsyncStatus
{
  public:
    AsyncStatus() = default;

    DELETE_COPYABILITY(AsyncStatus);
    DELETE_MOVEABILITY(AsyncStatus);

    Expected<ResponseT> await_response()
    {
        // todo(ryan): expand this into a wait_until with a deadline and a stop token
        auto event = m_promise.get_future().get();

        if (event.has_error())
        {
            return Error::create(event.error().message());
        }

        ResponseT response;
        if (!event.message().UnpackTo(&response))
        {
            throw Error::create("fatal error: unable to unpack message; server sent the wrong message type");
        }

        return response;
    }

  private:
    Promise<protos::Event> m_promise;
    friend Client;
};

template <typename ResponseT, typename RequestT>
Expected<ResponseT> Client::await_unary(const protos::EventType& event_type, RequestT&& request)
{
    AsyncStatus<ResponseT> status;
    async_unary(event_type, std::move(request), status);
    return status.await_response();
}

template <typename ResponseT, typename RequestT>
void Client::async_unary(const protos::EventType& event_type, RequestT&& request, AsyncStatus<ResponseT>& status)
{
    protos::Event event;
    event.set_event(event_type);
    event.set_tag(reinterpret_cast<std::uint64_t>(&status.m_promise));
    CHECK(event.mutable_message()->PackFrom(request));
    m_writer->await_write(std::move(event));
}

template <typename MessageT>
void Client::issue_event(const protos::EventType& event_type, MessageT&& message)
{
    protos::Event event;
    event.set_event(event_type);
    CHECK(event.mutable_message()->PackFrom(message));
    m_writer->await_write(std::move(event));
}

}  // namespace srf::internal::control_plane

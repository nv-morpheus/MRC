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

#include "internal/control_plane/client/connections_manager.hpp"
#include "internal/control_plane/client/instance.hpp"
#include "internal/control_plane/client/subscription_service.hpp"
#include "internal/expected.hpp"
#include "internal/grpc/client_streaming.hpp"
#include "internal/grpc/stream_writer.hpp"
#include "internal/network/resources.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/service.hpp"
#include "internal/ucx/resources.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/protos/architect.grpc.pb.h"
#include "mrc/protos/architect.pb.h"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/runner.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/macros.hpp"

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>
#include <grpcpp/channel.h>
#include <grpcpp/completion_queue.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mrc::internal::control_plane {

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

// todo: client should be a holder of the stream (private) and the connection manager (public)

class Client final : public resources::PartitionResourceBase, public Service
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

    using stream_t         = std::shared_ptr<rpc::ClientStream<mrc::protos::Event, mrc::protos::Event>>;
    using writer_t         = std::shared_ptr<rpc::StreamWriter<mrc::protos::Event>>;
    using event_t          = stream_t::element_type::IncomingData;
    using update_channel_t = mrc::node::SourceChannelWriteable<protos::StateUpdate>;

    Client(resources::PartitionResourceBase& base);

    // if we already have an grpc progress engine running, we don't need run another, just use that cq
    Client(resources::PartitionResourceBase& base, std::shared_ptr<grpc::CompletionQueue> cq);

    ~Client() final;

    const State& state() const
    {
        return m_state;
    }

    // MachineID machine_id() const;
    // const std::vector<InstanceID>& instance_ids() const;

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

    const mrc::runnable::LaunchOptions& launch_options() const;

    client::ConnectionsManager& connections() const
    {
        CHECK(m_connections_manager);
        return *m_connections_manager;
    }

    // request that the server start an update
    void request_update();

  private:
    void route_state_update(std::uint64_t tag, protos::StateUpdate&& update);

    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;
    void do_handle_event(event_t&& event);

    void forward_state(State state);

    State m_state{State::Disconnected};

    // MachineID m_machine_id;
    // std::vector<InstanceID> m_instance_ids;
    // std::map<InstanceID, std::unique_ptr<update_channel_t>> m_update_channels;
    // std::map<InstanceID, std::shared_ptr<client::Instance>> m_instances;

    std::shared_ptr<grpc::CompletionQueue> m_cq;
    std::shared_ptr<grpc::Channel> m_channel;
    std::shared_ptr<mrc::protos::Architect::Stub> m_stub;

    // if true, then the following runners should not be null
    // if false, then the following runners must be null
    const bool m_owns_progress_engine;
    std::unique_ptr<mrc::runnable::Runner> m_progress_handler;
    std::unique_ptr<mrc::runnable::Runner> m_progress_engine;
    std::unique_ptr<mrc::runnable::Runner> m_event_handler;

    // std::map<std::string, std::unique_ptr<node::SourceChannelWriteable<protos::StateUpdate>>> m_update_channels;
    // std::unique_ptr<client::ConnectionsManager> m_connections_manager;
    // std::map<std::string, std::unique_ptr<client::SubscriptionService>> m_subscription_services;

    // connection manager - connected to the update channel
    std::unique_ptr<client::ConnectionsManager> m_connections_manager;

    // update channel
    std::unique_ptr<mrc::node::SourceChannelWriteable<const protos::StateUpdate>> m_connections_update_channel;
    std::map<InstanceID, mrc::node::SourceChannelWriteable<const protos::StateUpdate>> m_instance_update_channels;

    // Stream Context
    stream_t m_stream;

    // StreamWriter acquired from m_stream->await_init()
    // The customer destruction of this object will cause a gRPC WritesDone to be issued to the server.
    writer_t m_writer;

    mrc::runnable::LaunchOptions m_launch_options;

    std::mutex m_mutex;

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

}  // namespace mrc::internal::control_plane

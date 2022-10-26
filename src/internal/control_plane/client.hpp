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

#include "srf/channel/status.hpp"
#include "srf/node/channel_holder.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/node/writable_subject.hpp"
#include "srf/protos/architect.grpc.pb.h"
#include "srf/protos/architect.pb.h"
#include "srf/runnable/runner.hpp"
#include "srf/types.hpp"

#include <nvrpc/client/client_fiber_streaming.h>

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace srf::internal::control_plane {

class ArchitectRuntime;

class Client : private nvrpc::client::fiber::ClientStreaming<protos::Event, protos::Event>
{
  public:
    using StreamingClient = nvrpc::client::fiber::ClientStreaming<protos::Event, protos::Event>;  // NOLINT

    Client(std::shared_ptr<ArchitectRuntime> runtime, std::shared_ptr<protos::Architect::Stub> stub);
    ~Client() override;

    const MachineID& machine_id() const
    {
        return m_machine_id;
    }

    const std::vector<InstanceID>& instance_ids() const
    {
        return m_instance_ids;
    }

    const std::string& info() const
    {
        return m_info;
    }

  protected:
    template <typename ResponseT, typename RequestT>
    ResponseT await_unary(const protos::EventType& event_type, RequestT&& request);

  private:
    void register_workers();

    // nvrpc::client::fiber::ClientStreaming virtual method
    std::unique_ptr<::grpc::ClientAsyncReaderWriter<protos::Event, protos::Event>> PrepareAsync(  // NOLINT
        ::grpc::ClientContext* context,
        ::grpc::CompletionQueue* cq) final;
    void TimeoutBackoff(const std::uint64_t& backoff) final;       // NOLINT
    void CallbackOnInitialized() final;                            // NOLINT
    void CallbackOnComplete(const ::grpc::Status& status) final;   // NOLINT
    void CallbackOnResponseReceived(protos::Event&& event) final;  // NOLINT

    std::shared_ptr<ArchitectRuntime> m_runtime;
    std::shared_ptr<protos::Architect::Stub> m_stub;
    std::unique_ptr<runnable::Runner> m_grpc_progress_engine;
    std::unique_ptr<runnable::Runner> m_event_handler_update_assignments;

    MachineID m_machine_id;
    std::vector<InstanceID> m_instance_ids;
    std::string m_info{"streaming_client: uninitialized"};

    std::unique_ptr<node::WritableSubject<protos::Event>> m_event_channel;

    // Completed by CallbackOnInitialized; indicates the bi-directional grpc is live
    Promise<void> m_promise_live;
    Promise<grpc::Status> m_promise_complete;
};

template <typename ResponseT, typename RequestT>
ResponseT Client::await_unary(const protos::EventType& event_type, RequestT&& request)
{
    ResponseT response;
    Promise<protos::Event> promise;
    auto future = promise.get_future();

    protos::Event event;
    event.set_machine_id(machine_id());
    event.set_event(event_type);
    event.set_promise(reinterpret_cast<std::uint64_t>(&promise));
    CHECK(event.mutable_message()->PackFrom(request));

    StreamingClient::Write(std::move(event));

    // get event from future, then check for status, if ok, return response
    CHECK(future.get().message().UnpackTo(&response));
    return response;
}

}  // namespace srf::internal::control_plane

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

#include "internal/control_plane/server.hpp"

#include "internal/control_plane/server_resources.hpp"

#include "srf/protos/architect.grpc.pb.h"
#include "srf/protos/architect.pb.h"

#include <nvrpc/context.h>
#include <nvrpc/executor.h>
#include <nvrpc/interfaces.h>
#include <nvrpc/life_cycle_streaming.h>
#include <nvrpc/rpc.h>
#include <nvrpc/server.h>
#include <nvrpc/service.h>

#include <glog/logging.h>
#include <google/protobuf/any.pb.h>

#include <algorithm>
#include <functional>
#include <ostream>
#include <type_traits>  // IWYU pragma: keep
#include <utility>

using nvrpc::AsyncRPC;
using nvrpc::AsyncService;
using nvrpc::Context;
using nvrpc::StreamingContext;

namespace srf::internal::control_plane {

class EventsStreamingContext final : public StreamingContext<protos::Event, protos::Event, ServerResources>
{
    void StreamInitialized(std::shared_ptr<ServerStream> stream) final  // NOLINT
    {
        // we don't have the instance id of the connecting stream, so we need to wait for
        // the client's first event to introduce itself and associated the server_stream_t
        // object with an instance_id
        VLOG(1) << "initializing events stream " << stream.get();
    }

    void RequestReceived(protos::Event&& request, std::shared_ptr<ServerStream> stream) final  // NOLINT
    {
        protos::Event response;

        VLOG(10) << "Event Received";

        switch (request.event())
        {
            // these events will be offloaded and handled by the oracle
            // these events will trigger a state update
            // these events are batched - all enqueued events will be processed
            // before a global state update is issued.

        case protos::EventType::ClientRegisterPipeline:
        case protos::EventType::ControlStop:
        case protos::EventType::ClientSegmentsOnComplete:
            DCHECK(GetResources()->validate_machine_id(request.machine_id(), stream.get())) << "invalid machine id";
            GetResources()->enqueue_event(std::move(request));
            break;

            // the remainder of events will be handled directly by the grpc event thread

            // the following are one sided message
            // response may be issued by the methods
            // these methods / events are used to synchronize clients when updates are issued

        case protos::EventType::ClientUpdateStart:
            GetResources()->on_client_update_start(std::move(request));
            break;
        case protos::EventType::ClientUpdateComplete:
            GetResources()->on_client_update_complete(std::move(request));
            break;

            // the following will be handled directly and a response issued

            // todo(ryan) - move the issue response into the resources method
            // this will move the response message, packing and issuing out of the switch

        case protos::EventType::ClientRegisterWorkers: {
            protos::RegisterWorkersRequest connection_request;
            protos::RegisterWorkersResponse connection_response;
            CHECK(request.message().UnpackTo(&connection_request));
            CHECK(GetResources()->register_workers(connection_request, connection_response, stream));
            request.set_machine_id(connection_response.machine_id());
            GetResources()->issue_response(request, std::move(connection_response));
        }
        break;

        case protos::ClientLookupWorkerAddresses: {
            protos::LookupWorkersRequest lookup_request;
            protos::LookupWorkersResponse lookup_response;
            CHECK(request.message().UnpackTo(&lookup_request));
            CHECK(GetResources()->lookup_workers(lookup_request, lookup_response));
            GetResources()->issue_response(request, std::move(lookup_response));
        }
        break;

        default:
            LOG(WARNING) << "Unknowned streaming event provided";
        }
    }

    void RequestsFinished(std::shared_ptr<ServerStream> stream) final  // NOLINT
    {
        // todo(ryan) - validate the machine has no assignmnts, then nothing further needs to be done
        // otherwise, this close writes signal is from a machine that still has assigned segments, in
        // which we need to update the other machines still working. this machine is lost to us, we can
        // still send it events, but it will be unable to respond.
        GetResources()->remove_machine(stream.get());
        stream->FinishStream();
    }

    void StreamFinished(std::shared_ptr<ServerStream> stream) final {}  // NOLINT
};

// Server

// todo(ryan) - this fiber executor is not working correctly; it is being instantiated as part of the server, but
// not used. currently using a standard thread executor
// todo(ryan) - convert this to a runnable

/*
class FiberExecutor final : public ::nvrpc::Worker
{
  public:
    FiberExecutor(std::size_t cq_count, RoundRobinFiberPool& pool) : ::nvrpc::Worker(cq_count), m_pool(pool) {}
    ~FiberExecutor() final
    {
        Shutdown();
    }

    void Run() final
    {
        CHECK_LE(m_pool.thread_count(), Size());
        for (int i = 0; i < Size(); ++i)
        {
            m_futures.push_back(m_pool.enqueue(FiberMetaData{INT32_MAX - 2}, [this, i] { ProgressEngine(i); }));
        }
    }

    void Shutdown() final
    {
        Worker::Shutdown();
        Join();
    }

  protected:
    bool IsAsync() const final
    {
        return true;
    }

    void TimeoutBackoff(std::uint64_t backoff) final
    {
        if (backoff < 16384)
        {
            boost::this_fiber::yield();
        }
        else
        {
            auto deadline = std::chrono::high_resolution_clock::now() + std::chrono::nanoseconds(backoff);
            boost::this_fiber::sleep_until(deadline);
        }
    }

    void Join()  // NOLINT
    {
        for (auto& f : m_futures)
        {
            f.get();
        }
    }

  private:
    std::vector<Future<void>> m_futures;
    RoundRobinFiberPool& m_pool;
};
*/

Server::Server(int port) : Server(std::string("0.0.0.0:") + std::to_string(port)) {}

Server::Server(std::string url) : m_server(std::make_unique<nvrpc::Server>(url))
{
    CHECK(m_server);
    auto* executor  = m_server->RegisterExecutor(new nvrpc::Executor(1));
    auto* architect = m_server->RegisterAsyncService<protos::Architect>();
    m_resources     = std::make_shared<ServerResources>();
    auto* rpc_event_stream =
        architect->RegisterRPC<EventsStreamingContext>(&protos::Architect::AsyncService::RequestEventStream);
    executor->RegisterContexts(rpc_event_stream, m_resources, 2);
    m_server->AsyncStart();
}

void Server::shutdown()
{
    m_resources->shutdown();
    m_server->Shutdown();
}
}  // namespace srf::internal::control_plane

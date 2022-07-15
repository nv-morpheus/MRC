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

#include "rxcpp/rx-subscriber.hpp"

#include "internal/control_plane/server_resources.hpp"
#include "internal/utils/contains.hpp"

#include "srf/node/edge_builder.hpp"
#include "srf/node/generic_node.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/protos/architect.grpc.pb.h"
#include "srf/protos/architect.pb.h"
#include "srf/runnable/launch_options.hpp"

#include <glog/logging.h>
#include <google/protobuf/any.pb.h>

#include <algorithm>
#include <functional>
#include <ostream>
#include <type_traits>  // IWYU pragma: keep
#include <utility>

namespace srf::internal::control_plane {

struct Server::Instance
{
    writer_t stream_writer;
    std::string worker_address;

    std::size_t get_id() const
    {
        return reinterpret_cast<std::size_t>(this);
    }
};

Server::Server(runnable::Resources& runnable) : m_runnable(runnable), m_server(m_runnable) {}

void Server::do_service_start()
{
    // create runnables
    auto acceptor = std::make_unique<srf::node::RxSource<stream_t>>(
        rxcpp::observable<>::create<stream_t>([this](rxcpp::subscriber<stream_t>& s) { do_accept_stream(s); }));

    m_queue = std::make_unique<srf::node::Queue<event_t>>();
    m_queue->enable_persistence();

    auto handler =
        std::make_unique<srf::node::RxSink<event_t>>([this](event_t event) { do_handle_event(std::move(event)); });

    // for edges between runnables
    srf::node::make_edge(*m_queue, *handler);

    // grpc service
    m_service = std::make_shared<srf::protos::Architect::AsyncService>();

    // bring up the grpc server and the progress engine
    m_server.register_service(m_service);
    m_server.service_start();

    // start the handler
    // if required, this is the runnable which most users would want to increase the level of concurrency
    // srf::runnable::LaunchOptions options;
    // options.engine_factory_name = "default";
    // options.pe_count = N;       // number of thread/cores
    // options.engines_per_pe = M; // number of fibers/user-threads per thread/core
    m_event_handler = m_runnable.launch_control().prepare_launcher(std::move(handler))->ignition();

    // start the acceptor - this should be one of the last runnables launch
    // once this goes live, connections will be accepted and data/events can be coming in
    m_stream_acceptor = m_runnable.launch_control().prepare_launcher(std::move(acceptor))->ignition();
}

void Server::do_service_await_live()
{
    m_server.service_await_live();
    m_event_handler->await_live();
    m_stream_acceptor->await_live();
}

void Server::do_service_stop()
{
    // if we are stopping the control plane and we are not in HA mode,
    // then all connections will be shutdown
    // to gracefully shutdown connections, we need to alert all services to go in to shutdown
    // mode which requires communication back and forth to the control, so we should not just
    // shutdown the server and the cq immeditately.
    // this is future work, for now we will be hard killing the server which will be hard killing the streams, the
    // clients will not gracefully shutdown and enter a kill mode.
    service_kill();
}

void Server::do_service_kill()
{
    // this is a hard stop, we are shutting everything down in the proper sequence to ensure clients get the kill
    // signal.

    // shutdown server and cqs
    m_server.service_kill();

    // clear all instances which drops their held stream writers
    m_instances.clear();

    // await all streams
    for (auto& [id, stream] : m_streams)
    {
        stream->await_fini();
    }

    // we keep the event handlers open until the streams are closed
    m_stream_acceptor->kill();
    m_queue->disable_persistence();
}

void Server::do_service_await_join()
{
    DVLOG(10) << "awaiting grpc server join";
    m_server.service_await_join();
    DVLOG(10) << "awaiting acceptor join";
    m_stream_acceptor->await_join();
    DVLOG(10) << "awaiting event handler join";
    m_event_handler->await_join();
    DVLOG(10) << "finished await_join";
}

/**
 * @brief Stream Acceptor
 *
 * The while loop of this method says active as long as the grpc server is still accepting connections.
 * There are multiple way this can be implemented depending the service requirements, one might choose
 * to preallocate N number of streams and issues them all to the CQ. This is an alternative method which
 * creates a single stream and waits for it to get initialized, then creates another. The current implementation is
 * unbounded an upper bound could be added.
 *
 * This method works well for the requirements of the SRF control plane where the number of connections is relatively
 * small and the duration of the connection is long.
 */
void Server::do_accept_stream(rxcpp::subscriber<stream_t>& s)
{
    auto cq = m_server.get_cq();

    auto request_fn = [this, cq](grpc::ServerContext* context,
                                 grpc::ServerAsyncReaderWriter<srf::protos::Event, srf::protos::Event>* stream,
                                 void* tag) {
        m_service->RequestEventStream(context, stream, cq.get(), cq.get(), tag);
    };

    while (s.is_subscribed())
    {
        // create stream
        auto stream = std::make_shared<typename stream_t::element_type>(request_fn, m_runnable);

        // attach to handler
        stream->attach_to(*m_queue);

        // await for incoming connection
        auto writer = stream->await_init();

        if (!writer)
        {
            // the server is shutting down
            break;
        }

        // save new stream
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        CHECK_EQ(stream->get_id(), writer->get_id());
        auto search = m_streams.find(stream->get_id());
        CHECK(search == m_streams.end());
        m_streams[stream->get_id()] = stream;
    }

    // await and finish all streams
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    for (auto& [ptr, stream] : m_streams)
    {
        stream->await_fini();
    }

    s.on_completed();
}

void Server::do_handle_event(event_t&& event)
{
    DCHECK(event.stream);

    if (event.ok)
    {
        switch (event.msg.event())
        {
        case protos::EventType::ClientRegisterWorkers:
            register_workers(event);
            break;
        default:
            LOG(FATAL) << "event not handled";
        }
    }
    else
    {
        drop_stream(event.stream);
    }
}

void Server::register_workers(event_t& event)
{
    protos::RegisterWorkersRequest req;
    protos::RegisterWorkersResponse resp;
    CHECK(event.msg.message().UnpackTo(&req));

    DVLOG(10) << "registering srf instance with " << req.ucx_worker_addresses_size() << " resource groups";

    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    // validate that the worker addresses are valid before updating state
    for (const auto& worker_address : req.ucx_worker_addresses())
    {
        auto search = m_ucx_worker_addresses.find(worker_address);
        if (search != m_ucx_worker_addresses.end())
        {
            LOG(FATAL) << "received a connection request for a ucx worker that is already registered";
        }
    }

    // set machine id for the current stream
    resp.set_machine_id(event.stream->get_id());

    for (const auto& worker_address : req.ucx_worker_addresses())
    {
        m_ucx_worker_addresses.insert(worker_address);

        auto instance            = std::make_shared<Instance>();
        instance->worker_address = worker_address;
        instance->stream_writer  = event.stream;

        CHECK(!contains(m_instances, instance->get_id()));
        m_instances[instance->get_id()] = instance;
        m_instances_by_stream.insert(std::pair{event.stream->get_id(), instance->get_id()});

        DVLOG(10) << "adding instance id to response: " << instance->get_id()
                  << " assigned to unique node: " << event.stream->get_id();

        // return in order provided a unique instance_id per partition ucx address
        resp.add_instance_ids(instance->get_id());
    }
}

void Server::drop_stream(writer_t writer)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    // the associated srf instance associated with this stream should be fully disconnected from all services
    // auto count = connected_services(event.stream->get_id());
    // if (count)
    // {
    //     // this is an unexpected disconnect
    //     // we need to detatch the currenet stream from every service
    //     // this may be a fatal condition
    //     on_unexpected_disconnect(event.stream);
    // }

    // find all instances associated with the writer's stream
    auto instances = m_instances_by_stream.equal_range(writer->get_id());

    // drop all instances
    for (auto i = instances.first; i != instances.second; ++i)
    {
        auto search = m_instances.find(i->second);
        CHECK(search != m_instances.end());
        m_instances.erase(search);
    }

    // remove the mapping from stream_id -> instance_ids
    m_instances_by_stream.erase(writer->get_id());

    // get stream / stream context
    auto stream = m_streams.find(writer->get_id());
    CHECK(stream != m_streams.end());

    // drop the last writer which is holding the response stream open
    // CHECK_EQ(writer.use_count(), 1);
    writer->finish();
    writer.reset();

    // await completion of the stream connection
    stream->second->await_fini();
    m_streams.erase(stream);
}

// class EventsStreamingContext final : public StreamingContext<protos::Event, protos::Event, ServerResources>
// {
//     void StreamInitialized(std::shared_ptr<ServerStream> stream) final  // NOLINT
//     {
//         // we don't have the instance id of the connecting stream, so we need to wait for
//         // the client's first event to introduce itself and associated the server_stream_t
//         // object with an instance_id
//         VLOG(1) << "initializing events stream " << stream.get();
//     }

//     void RequestReceived(protos::Event&& request, std::shared_ptr<ServerStream> stream) final  // NOLINT
//     {
//         protos::Event response;

//         VLOG(10) << "Event Received";

//         switch (request.event())
//         {
//             // these events will be offloaded and handled by the oracle
//             // these events will trigger a state update
//             // these events are batched - all enqueued events will be processed
//             // before a global state update is issued.

//         case protos::EventType::ClientRegisterPipeline:
//         case protos::EventType::ControlStop:
//         case protos::EventType::ClientSegmentsOnComplete:
//             DCHECK(GetResources()->validate_machine_id(request.machine_id(), stream.get())) << "invalid machine id";
//             GetResources()->enqueue_event(std::move(request));
//             break;

//             // the remainder of events will be handled directly by the grpc event thread

//             // the following are one sided message
//             // response may be issued by the methods
//             // these methods / events are used to synchronize clients when updates are issued

//         case protos::EventType::ClientUpdateStart:
//             GetResources()->on_client_update_start(std::move(request));
//             break;
//         case protos::EventType::ClientUpdateComplete:
//             GetResources()->on_client_update_complete(std::move(request));
//             break;

//             // the following will be handled directly and a response issued

//             // todo(ryan) - move the issue response into the resources method
//             // this will move the response message, packing and issuing out of the switch

//         case protos::EventType::ClientRegisterWorkers: {
//             protos::RegisterWorkersRequest connection_request;
//             protos::RegisterWorkersResponse connection_response;
//             CHECK(request.message().UnpackTo(&connection_request));
//             CHECK(GetResources()->register_workers(connection_request, connection_response, stream));
//             request.set_machine_id(connection_response.machine_id());
//             GetResources()->issue_response(request, std::move(connection_response));
//         }
//         break;

//         case protos::ClientLookupWorkerAddresses: {
//             protos::LookupWorkersRequest lookup_request;
//             protos::LookupWorkersResponse lookup_response;
//             CHECK(request.message().UnpackTo(&lookup_request));
//             CHECK(GetResources()->lookup_workers(lookup_request, lookup_response));
//             GetResources()->issue_response(request, std::move(lookup_response));
//         }
//         break;

//         default:
//             LOG(WARNING) << "Unknowned streaming event provided";
//         }
//     }

//     void RequestsFinished(std::shared_ptr<ServerStream> stream) final  // NOLINT
//     {
//         // todo(ryan) - validate the machine has no assignmnts, then nothing further needs to be done
//         // otherwise, this close writes signal is from a machine that still has assigned segments, in
//         // which we need to update the other machines still working. this machine is lost to us, we can
//         // still send it events, but it will be unable to respond.
//         GetResources()->remove_machine(stream.get());
//         stream->FinishStream();
//     }

//     void StreamFinished(std::shared_ptr<ServerStream> stream) final {}  // NOLINT
// };

// Server::Server(int port) : Server(std::string("0.0.0.0:") + std::to_string(port)) {}

// Server::Server(std::string url) : m_server(std::make_unique<nvrpc::Server>(url))
// {
//     CHECK(m_server);
//     auto* executor  = m_server->RegisterExecutor(new nvrpc::Executor(1));
//     auto* architect = m_server->RegisterAsyncService<protos::Architect>();
//     m_resources     = std::make_shared<ServerResources>();
//     auto* rpc_event_stream =
//         architect->RegisterRPC<EventsStreamingContext>(&protos::Architect::AsyncService::RequestEventStream);
//     executor->RegisterContexts(rpc_event_stream, m_resources, 2);
//     m_server->AsyncStart();
// }

// void Server::shutdown()
// {
//     m_resources->shutdown();
//     m_server->Shutdown();
// }

}  // namespace srf::internal::control_plane

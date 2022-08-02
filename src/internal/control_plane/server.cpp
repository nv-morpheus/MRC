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
#include "tl/expected.hpp"

#include "internal/control_plane/server_resources.hpp"
#include "internal/utils/contains.hpp"

#include "srf/channel/status.hpp"
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

#define SRF_EXPECT_TRUE(expected)                          \
    if (!expected)                                         \
    {                                                      \
        return Error::create(std::move(expected.error())); \
    }

namespace srf::internal::control_plane {

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
    Expected<> status;

    if (event.ok)
    {
        switch (event.msg.event())
        {
        case protos::EventType::ClientUnaryRegisterWorkers:
            status = unary_response(event, unary_register_workers(event));
            break;

        case protos::EventType::ClientUnaryCreateSubscriptionService:
            status = unary_response(event, unary_create_subscription_service(event));
            break;

        case protos::EventType::ClientUnaryRegisterSubscriptionService:
            status = unary_response(event, unary_register_subscription_service(event));
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

Expected<protos::RegisterWorkersResponse> Server::unary_register_workers(event_t& event)
{
    auto req = unpack_request<protos::RegisterWorkersRequest>(event);
    SRF_EXPECT_TRUE(req);

    DVLOG(10) << "registering srf instance with " << req->ucx_worker_addresses_size() << " partitions groups";

    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    // validate that the worker addresses are valid before updating state
    for (const auto& worker_address : req->ucx_worker_addresses())
    {
        auto search = m_ucx_worker_addresses.find(worker_address);
        if (search != m_ucx_worker_addresses.end())
        {
            Error::create("invalid ucx worker address(es) - duplicate registration(s) detected");
        }
    }

    // set machine id for the current stream
    protos::RegisterWorkersResponse resp;
    resp.set_machine_id(event.stream->get_id());

    for (const auto& worker_address : req->ucx_worker_addresses())
    {
        m_ucx_worker_addresses.insert(worker_address);

        auto instance            = std::make_shared<server::ClientInstance>();
        instance->worker_address = worker_address;
        instance->stream_writer  = event.stream;

        if (contains(m_instances, instance->get_id()))
        {
            throw Error::create("non-unique instance_id detected");
        }
        m_instances[instance->get_id()] = instance;
        m_instances_by_stream.insert(std::pair{event.stream->get_id(), instance->get_id()});

        DVLOG(10) << "adding instance id to response: " << instance->get_id()
                  << " assigned to unique node: " << event.stream->get_id();

        // return in order provided a unique instance_id per partition ucx address
        resp.add_instance_ids(instance->get_id());
    }

    return resp;
}

Expected<protos::Ack> Server::unary_create_subscription_service(event_t& event)
{
    auto req = unpack_request<protos::CreateSubscriptionServiceRequest>(event);
    SRF_EXPECT_TRUE(req);

    DVLOG(10) << "[start] create (or get) subscription service: " << req->service_name();

    std::set<std::string> roles;
    for (const auto& role : req->roles())
    {
        roles.insert(role);
    }
    if (roles.size() != req->roles_size())
    {
        return Error::create("duplicate roles detected");
    }

    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    auto search = m_subscription_services.find(req->service_name());
    if (search == m_subscription_services.end())
    {
        DVLOG(10) << "subscription_service: " << req->service_name()
                  << " first request - creating subscription service";
        m_subscription_services[req->service_name()] =
            std::make_unique<SubscriptionService>(req->service_name(), std::move(roles));
    }
    else
    {
        if (!search->second->compare_roles(roles))
        {
            std::stringstream msg;
            msg << "failed to create subscription service on the server: requested roles do not match the current "
                   "instance of "
                << req->service_name()
                << "; there may be a binary incompatibililty or service name conflict between one or more clients "
                   "connecting to this control plane";

            return Error::create(msg.str());
        }
    }

    DVLOG(10) << "[success] create (or get) subscription service: " << req->service_name();
    return ack_success();
}

Expected<protos::Ack> Server::unary_register_subscription_service(event_t& event)
{
    auto req = unpack_request<protos::RegisterSubscriptionServiceRequest>(event);
    SRF_EXPECT_TRUE(req);

    // validate message - can be done before locking internal state
    auto subscribe_to = check_unique_repeated_field(req->subscribe_to_roles());
    SRF_EXPECT_TRUE(subscribe_to);

    // lock internal state
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    DVLOG(10) << "[start] instance_id: [id]; register with subscription service " << req->service_name() << " as a "
              << req->role() << " from machine " << event.stream->get_id();

    auto instance = validate_instance_id(req->instance_id(), event);
    SRF_EXPECT_TRUE(instance);

    auto service_iter = get_subscription_service(req->service_name());
    SRF_EXPECT_TRUE(service_iter);
    auto& service = *(service_iter.value()->second);

    // validate roles are valid
    if (!service.has_role(req->role()))
    {
        return Error::create(SRF_CONCAT_STR(
            "subscription service " << req->service_name() << " does not contain primary role: " << req->role()));
    }
    if (!std::all_of(subscribe_to.value().begin(), subscribe_to.value().end(), [&service](const std::string& role) {
            return service.has_role(role);
        }))
    {
        return Error::create(SRF_CONCAT_STR("subscription service " << req->service_name()
                                                                    << " one or more subscribe_to_roles were invalid"));
    }

    // todo(ryan) - need improved error handling
    service.register_instance(*instance, req->role(), *subscribe_to);

    DVLOG(10) << "[success] register subscription service: " << req->service_name() << "; role: " << req->role();
    return ack_success();
}

Expected<protos::Ack> Server::unary_drop_from_subscription_service(event_t& event)
{
    auto req = unpack_request<protos::SubscriptionServiceUpdate>(event);
    SRF_EXPECT_TRUE(req);

    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    auto valid_ids = std::all_of(req->instance_ids().begin(),
                                 req->instance_ids().end(),
                                 [this, &event](const instance_id_t& id) { return validate_instance_id(id, event); });

    auto search        = m_subscription_services.find(req->service_name());
    auto valid_service = (search != m_subscription_services.end());

    if (valid_ids && valid_service)
    {
        CHECK(search->second);
        for (const auto& id : req->instance_ids())
        {
            auto instance = get_instance(id);
            search->second->drop_instance(*instance);
        }
    }

    return ack_success();
}

void Server::drop_stream(writer_t writer)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    // the srf instance associated with this stream should be fully disconnected from all services
    // auto count = connected_services(event.stream->get_id());
    // if (count)
    // {
    //     // this is an unexpected disconnect
    //     // we need to detatch the currenet stream from every service
    //     // this may be a fatal condition
    //     on_unexpected_disconnect(event.stream);
    // }

    DVLOG(10) << "dropping stream with machine_id: " << writer->get_id();

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

Expected<> Server::unary_ack(event_t& event, protos::ErrorCode type, std::string msg)
{
    protos::Ack ack;
    ack.set_status(type);
    ack.set_msg(msg);

    protos::Event error;
    error.set_event(protos::EventType::Response);
    error.set_tag(event.msg.tag());
    CHECK(error.mutable_message()->PackFrom(ack));
    if (event.stream->await_write(std::move(error)) != channel::Status::success)
    {
        return Error::create("failed to write to channel");
    }
    return {};
}

Expected<Server::instance_t> Server::validate_instance_id(const instance_id_t& instance_id, const event_t& event) const
{
    return get_instance(instance_id).and_then([&event, &instance_id](auto& i) -> Expected<instance_t> {
        if (event.stream->get_id() != i->stream_writer->get_id())
        {
            return Error::create(SRF_CONCAT_STR(
                "instance_id (" << instance_id << ") not assocated with machine/stream: " << event.stream->get_id()));
        }
        return i;
    });
}

Expected<Server::instance_t> Server::get_instance(const instance_id_t& instance_id) const
{
    auto search = m_instances.find(instance_id);
    if (search == m_instances.end())
    {
        return Error::create("invalid instance_id");
    }
    return search->second;
}

Expected<decltype(Server::m_subscription_services)::const_iterator> Server::get_subscription_service(
    const std::string& name) const
{
    auto search = m_subscription_services.find(name);
    if (search == m_subscription_services.end())
    {
        return Error::create("invalid subscription_service name");
    }
    return search;
}

protos::Ack Server::ack_success()
{
    protos::Ack ack;
    ack.set_status(protos::Success);
    return ack;
}

}  // namespace srf::internal::control_plane

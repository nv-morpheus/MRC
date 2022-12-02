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

#include "internal/control_plane/proto_helpers.hpp"
#include "internal/control_plane/server/subscription_manager.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/protos/architect.grpc.pb.h"
#include "mrc/protos/architect.pb.h"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launcher.hpp"

#include <boost/fiber/condition_variable.hpp>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <rxcpp/sources/rx-iterate.hpp>

#include <algorithm>
#include <exception>
#include <future>
#include <mutex>
#include <ostream>
#include <set>
#include <utility>
#include <vector>

namespace mrc::internal::control_plane {

template <typename T>
static Expected<T> unpack_request(Server::event_t& event)
{
    if (event.msg.has_message())
    {
        return unpack<T>(event.msg.message());
    }
    if (event.msg.has_error())
    {
        return Error::create(event.msg.error().message());
    }
    return Error::create("client request has neither a message, nor an error - invalid request");
}

template <typename MessageT>
static Expected<> unary_response(Server::event_t& event, Expected<MessageT>&& message)
{
    if (!message)
    {
        protos::Error error;
        error.set_code(protos::ErrorCode::InstanceError);
        error.set_message(message.error().message());
        return unary_response<protos::Error>(event, std::move(error));
    }
    mrc::protos::Event out;
    out.set_tag(event.msg.tag());
    out.set_event(protos::EventType::Response);
    out.mutable_message()->PackFrom(*message);
    if (event.stream->await_write(std::move(out)) != channel::Status::success)
    {
        return Error::create("failed to write to channel");
    }
    return {};
}

Server::Server(runnable::Resources& runnable) : m_runnable(runnable), m_server(m_runnable) {}

void Server::do_service_start()
{
    // node to accept connections
    auto acceptor = std::make_unique<mrc::node::RxSource<stream_t>>(
        rxcpp::observable<>::create<stream_t>([this](rxcpp::subscriber<stream_t>& s) { do_accept_stream(s); }));

    // node to periodically issue updates

    // create external queue for incoming events
    // as new grpc streams are initialized by the acceptor, they attach as sources to the queue (stream >> queue)
    // these streams issue event (event_t) object which encapsulate the stream_writer for the originating stream
    m_queue = std::make_unique<mrc::node::Queue<event_t>>();
    m_queue->enable_persistence();

    // the queue is attached to the event handler which will update the internal state of the server
    auto handler =
        std::make_unique<mrc::node::RxSink<event_t>>([this](event_t event) { do_handle_event(std::move(event)); });

    // node to periodically issue update of the server state to connected clients via the grpc bidi streams
    auto updater = std::make_unique<mrc::node::RxSource<void*>>(
        rxcpp::observable<>::create<void*>([this](rxcpp::subscriber<void*>& s) { do_issue_update(s); }));

    // edge: queue >> handler
    mrc::node::make_edge(*m_queue, *handler);

    // grpc service
    m_service = std::make_shared<mrc::protos::Architect::AsyncService>();

    // bring up the grpc server and the progress engine
    m_server.register_service(m_service);
    m_server.service_start();

    // start the handler
    // if required, this is the runnable which most users would want to increase the level of concurrency
    // mrc::runnable::LaunchOptions options;
    // options.engine_factory_name = "default";
    // options.pe_count = N;       // number of thread/cores
    // options.engines_per_pe = M; // number of fibers/user-threads per thread/core
    m_event_handler = m_runnable.launch_control().prepare_launcher(std::move(handler))->ignition();

    // periodic updater
    m_update_handler = m_runnable.launch_control().prepare_launcher(std::move(updater))->ignition();

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
    m_stream_acceptor->stop();
    m_update_handler->stop();
    m_update_cv.notify_all();

    service_kill();
}

void Server::do_service_kill()
{
    // this is a hard stop, we are shutting everything down in the proper sequence to ensure clients get the kill
    // signal.
    m_stream_acceptor->kill();
    m_update_handler->kill();
    m_update_cv.notify_all();

    // shutdown server and cqs
    m_server.service_kill();
}

void Server::do_service_await_join()
{
    // clear all instances which drops their held stream writers
    DVLOG(10) << "awaiting all streams";
    m_connections.drop_all_streams();

    // we keep the event handlers open until the streams are closed
    m_queue->disable_persistence();

    DVLOG(10) << "awaiting grpc server join";
    m_server.service_await_join();
    DVLOG(10) << "awaiting acceptor join";
    m_stream_acceptor->await_join();
    DVLOG(10) << "awaiting updater join";
    m_update_handler->await_join();
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
 * This method works well for the requirements of the MRC control plane where the number of connections is relatively
 * small and the duration of the connection is long.
 */
void Server::do_accept_stream(rxcpp::subscriber<stream_t>& s)
{
    auto cq = m_server.get_cq();

    auto request_fn = [this, cq](grpc::ServerContext* context,
                                 grpc::ServerAsyncReaderWriter<mrc::protos::Event, mrc::protos::Event>* stream,
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

        // contract validation
        DCHECK_EQ(stream->get_id(), writer->get_id());

        // save new stream
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        m_connections.add_stream(stream);
    }

    s.on_completed();
}

void Server::do_handle_event(event_t&& event)
{
    DCHECK(event.stream);

    try
    {
        if (event.ok)
        {
            Expected<> status;
            switch (event.msg.event())
            {
            case protos::EventType::ClientEventRequestStateUpdate:
                DVLOG(10) << "client requested a server update";
                // todo: add a backoff so if a bunch of clients issue update requests
                // we don't just keep firing them server side
                m_update_cv.notify_one();
                break;

            case protos::EventType::ClientUnaryRegisterWorkers:
                status = unary_register_workers(event);
                break;

            case protos::EventType::ClientUnaryActivateStream:
                status = unary_activate_stream(event);
                break;

            case protos::EventType::ClientUnaryLookupWorkerAddresses:
                status = unary_lookup_workers(event);
                break;

            case protos::EventType::ClientUnaryDropWorker:
                status = unary_drop_worker(event);
                break;

            case protos::EventType::ClientUnaryCreateSubscriptionService:
                status = unary_response(event, unary_create_subscription_service(event));
                break;

            case protos::EventType::ClientUnaryRegisterSubscriptionService:
                status = unary_response(event, unary_register_subscription_service(event));
                break;

            case protos::EventType::ClientUnaryActivateSubscriptionService:
                status = unary_response(event, unary_activate_subscription_service(event));
                break;

            case protos::EventType::ClientUnaryDropSubscriptionService:
                status = unary_response(event, unary_drop_subscription_service(event));
                break;

            case protos::EventType::ClientEventUpdateSubscriptionService:
                status = event_update_subscription_service(event);
                break;

            default:
                LOG(ERROR) << "unhandled event type in server handler";
                throw Error::create("unhandled event type in server handler");
            }

            if (!status)
            {
                throw status.error();
            }
        }
        else
        {
            drop_stream(event.stream);
        }
    } catch (const tl::bad_expected_access<Error>& e)
    {
        LOG(ERROR) << "bad_expected_access: " << e.error().message();
        on_fatal_exception();
    } catch (const UnexpectedError& e)
    {
        LOG(ERROR) << "unexpected: " << e.value().message();
        on_fatal_exception();
    } catch (const std::exception& e)
    {
        LOG(ERROR) << "exception: " << e.what();
        on_fatal_exception();
    } catch (...)
    {
        LOG(ERROR) << "unknown exception caught";
        on_fatal_exception();
    }
}

void Server::do_issue_update(rxcpp::subscriber<void*>& s)
{
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);

    for (;;)
    {
        auto status = m_update_cv.wait_for(lock, m_update_period);
        if (!s.is_subscribed())
        {
            s.on_completed();
            return;
        }

        DVLOG(10) << "starting - control plane update";

        // issue worker updates
        m_connections.issue_update();

        // issue subscription service updates
        for (auto& [name, service] : m_subscription_services)
        {
            service->issue_update();
        }

        DVLOG(10) << "finished - control plane update";
    }
}

void Server::on_fatal_exception()
{
    LOG(FATAL) << "fatal error on the control plane server was caught; signal all attached instances to shutdown "
                  "and disconnect";

    // todo: convert the FATAL to ERROR, then mark the server as shutting down, then issue shutdown requests
    // to each connected client, then close the client connections with a grpc CANCELLED on the steam.
    // the clients should receive the shutdown message with the understanding that the server will no longer be
    // responding to events. this means, the status objects used to hold a fiber promise should never fully block and
    // instead use a long deadline and a stop token which they must check if the deadline ever times out.
}

Expected<> Server::unary_register_workers(event_t& event)
{
    auto req = unpack_request<protos::RegisterWorkersRequest>(event);
    MRC_EXPECT(req);

    DVLOG(10) << "registering stream " << event.stream->get_id() << " with " << req->ucx_worker_addresses_size()
              << " partitions groups";
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return unary_response(event, m_connections.register_instances(event.stream, *req));
}

Expected<> Server::unary_drop_worker(event_t& event)
{
    auto req = unpack_request<protos::TaggedInstance>(event);
    MRC_EXPECT(req);

    DVLOG(10) << "dropping instance " << req->instance_id() << " from stream " << event.stream->get_id();
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    // ensure all server-side state machines have dropped the requested instance_id
    drop_instance(req->instance_id());

    // drop the instance id from the connection manager
    return unary_response(event, m_connections.drop_instance(event.stream, *req));
}

Expected<> Server::unary_activate_stream(event_t& event)
{
    auto message = unpack_request<protos::RegisterWorkersResponse>(event);
    MRC_EXPECT(message);
    DVLOG(10) << "activating stream " << message->machine_id() << " with " << message->instance_ids_size()
              << " instances/partitions";
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return unary_response(event, m_connections.activate_stream(event.stream, *message));
}

Expected<> Server::unary_lookup_workers(event_t& event)
{
    auto message = unpack_request<protos::LookupWorkersRequest>(event);
    MRC_EXPECT(message);
    DVLOG(10) << "looking up worker addresses for " << message->instance_ids_size() << " instances";
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return unary_response(event, m_connections.lookup_workers(event.stream, *message));
}

Expected<protos::Ack> Server::unary_create_subscription_service(event_t& event)
{
    auto req = unpack_request<protos::CreateSubscriptionServiceRequest>(event);
    MRC_EXPECT(req);

    DVLOG(10) << "[start] create (or get) subscription service: " << req->service_name();

    std::set<std::string> roles;
    for (const auto& role : req->roles())
    {
        roles.insert(role);
    }
    if (roles.size() != req->roles_size())
    {
        return Error::create("duplicate roles detected; all roles must have unique names");
    }

    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    auto search = m_subscription_services.find(req->service_name());
    if (search == m_subscription_services.end())
    {
        DVLOG(10) << "subscription_service: " << req->service_name()
                  << " first request - creating subscription service";
        m_subscription_services[req->service_name()] =
            std::make_unique<server::SubscriptionService>(req->service_name(), std::move(roles));
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
    return protos::Ack{};
}

Expected<protos::RegisterSubscriptionServiceResponse> Server::unary_register_subscription_service(event_t& event)
{
    auto req = unpack_request<protos::RegisterSubscriptionServiceRequest>(event);
    MRC_EXPECT(req);

    // validate message - can be done before locking internal state
    auto subscribe_to = check_unique_repeated_field(req->subscribe_to_roles());
    MRC_EXPECT(subscribe_to);

    // lock internal state
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    DVLOG(10) << "[start] register with subscription service " << req->service_name() << " as a " << req->role()
              << " from machine " << event.stream->get_id();

    auto instance = validate_instance_id(req->instance_id(), event);
    MRC_EXPECT(instance);

    auto service_iter = get_subscription_service(req->service_name());
    MRC_EXPECT(service_iter);
    auto& service = *(service_iter.value()->second);

    // validate roles are valid
    if (!service.has_role(req->role()))
    {
        return Error::create(MRC_CONCAT_STR(
            "subscription service " << req->service_name() << " does not contain primary role: " << req->role()));
    }
    if (!std::all_of(subscribe_to.value().begin(), subscribe_to.value().end(), [&service](const std::string& role) {
            return service.has_role(role);
        }))
    {
        return Error::create(MRC_CONCAT_STR("subscription service " << req->service_name()
                                                                    << " one or more subscribe_to_roles were invalid"));
    }

    auto tag = service.register_instance(*instance, req->role(), *subscribe_to);
    MRC_EXPECT(tag);

    DVLOG(10) << "[success] register subscription service: " << req->service_name() << "; role: " << req->role();
    protos::RegisterSubscriptionServiceResponse resp;
    resp.set_service_name(req->service_name());
    resp.set_role(req->role());
    resp.set_tag(*tag);
    return resp;
}

Expected<protos::Ack> Server::unary_activate_subscription_service(event_t& event)
{
    auto req = unpack_request<protos::ActivateSubscriptionServiceRequest>(event);
    MRC_EXPECT(req);

    // validate message - can be done before locking internal state
    auto subscribe_to = check_unique_repeated_field(req->subscribe_to_roles());
    MRC_EXPECT(subscribe_to);

    // lock internal state
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    DVLOG(10) << "[start] instance_id: [id]; activate with subscription service " << req->service_name() << " as a "
              << req->role() << " from machine " << event.stream->get_id();

    auto instance = validate_instance_id(req->instance_id(), event);
    MRC_EXPECT(instance);

    auto service_iter = get_subscription_service(req->service_name());
    MRC_EXPECT(service_iter);
    auto& service = *(service_iter.value()->second);

    // validate roles are valid
    if (!service.has_role(req->role()))
    {
        return Error::create(MRC_CONCAT_STR(
            "subscription service " << req->service_name() << " does not contain primary role: " << req->role()));
    }
    if (!std::all_of(subscribe_to.value().begin(), subscribe_to.value().end(), [&service](const std::string& role) {
            return service.has_role(role);
        }))
    {
        return Error::create(MRC_CONCAT_STR("subscription service " << req->service_name()
                                                                    << " one or more subscribe_to_roles were invalid"));
    }

    MRC_EXPECT(service.activate_instance(*instance, req->role(), *subscribe_to, req->tag()));
    DVLOG(10) << "[success] activate subscription service: " << req->service_name() << "; role: " << req->role();

    return {};
}

Expected<protos::Ack> Server::unary_drop_subscription_service(event_t& event)
{
    auto req = unpack_request<protos::DropSubscriptionServiceRequest>(event);
    MRC_EXPECT(req);

    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    auto instance = validate_instance_id(req->instance_id(), event);
    MRC_EXPECT(instance);

    auto service_iter = get_subscription_service(req->service_name());
    MRC_EXPECT(service_iter);
    auto& service = *(service_iter.value()->second);

    service.drop_tag(req->tag());
    return {};
}

Expected<> Server::event_update_subscription_service(event_t& event)
{
    auto req = unpack_request<protos::UpdateSubscriptionServiceRequest>(event);
    MRC_EXPECT(req);

    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    auto service_iter = get_subscription_service(req->service_name());
    MRC_EXPECT(service_iter);
    auto& service = *(service_iter.value()->second);

    return service.update_role(*req);
}

void Server::drop_instance(const instance_id_t& instance_id)
{
    // add any future state machine, e.g. pipeline, segment, manifold, etc. here
    for (auto& [service_name, service] : m_subscription_services)
    {
        service->drop_instance(instance_id);
    }
}

void Server::drop_stream(writer_t& writer)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    const auto stream_id = writer->get_id();
    DVLOG(10) << "dropping stream with machine_id: " << stream_id;

    // for each instance - iterate over state machines and drop the instance id
    for (const auto& instance_id : m_connections.get_instance_ids(stream_id))
    {
        drop_instance(instance_id);
    }

    // close stream - finish is a noop if the stream was previously cancelled
    writer->finish();
    writer.reset();

    m_connections.drop_stream(stream_id);
}

Expected<Server::instance_t> Server::validate_instance_id(const instance_id_t& instance_id, const event_t& event) const
{
    return m_connections.get_instance(instance_id).and_then([&event, &instance_id](auto& i) -> Expected<instance_t> {
        if (event.stream->get_id() != i->stream_writer().get_id())
        {
            return Error::create(MRC_CONCAT_STR(
                "instance_id (" << instance_id << ") not assocated with machine/stream: " << event.stream->get_id()));
        }
        return i;
    });
}

Expected<Server::instance_t> Server::get_instance(const instance_id_t& instance_id) const
{
    return m_connections.get_instance(instance_id);
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

}  // namespace mrc::internal::control_plane

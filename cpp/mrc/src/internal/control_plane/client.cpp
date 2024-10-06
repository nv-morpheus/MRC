/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/control_plane/client.hpp"

#include "internal/control_plane/client/connections_manager.hpp"
#include "internal/grpc/progress_engine.hpp"
#include "internal/grpc/promise_handler.hpp"  // for PromiseHandler
#include "internal/grpc/stream_writer.hpp"    // for StreamWriter
#include "internal/runnable/runnable_resources.hpp"
#include "internal/service.hpp"
#include "internal/system/system.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/edge/edge_builder.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/writable_entrypoint.hpp"
#include "mrc/options/options.hpp"
#include "mrc/protos/architect.grpc.pb.h"
#include "mrc/protos/architect.pb.h"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runnable/runner.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/promise.hpp>  // for promise
#include <google/protobuf/any.pb.h>
#include <grpcpp/grpcpp.h>
#include <rxcpp/rx.hpp>

#include <mutex>
#include <ostream>

namespace mrc::control_plane {

std::atomic_uint64_t AsyncEventStatus::s_request_id_counter;

AsyncEventStatus::AsyncEventStatus() : m_request_id(++s_request_id_counter) {}

size_t AsyncEventStatus::request_id() const
{
    return m_request_id;
}

void AsyncEventStatus::set_future(Future<protos::Event> future)
{
    m_future = std::move(future);
}

Client::Client(resources::PartitionResourceBase& base, std::shared_ptr<grpc::CompletionQueue> cq) :
  resources::PartitionResourceBase(base),
  Service("control_plane::Client"),
  m_cq(std::move(cq)),
  m_owns_progress_engine(false)
{}

Client::Client(resources::PartitionResourceBase& base) :
  resources::PartitionResourceBase(base),
  Service("control_plane::Client"),
  m_cq(std::make_shared<grpc::CompletionQueue>()),
  m_owns_progress_engine(true)
{}

Client::~Client()
{
    Service::call_in_destructor();
}

void Client::do_service_start()
{
    m_launch_options.engine_factory_name = "main";
    m_launch_options.pe_count            = 1;
    m_launch_options.engines_per_pe      = 1;

    auto url = runnable().system().options().architect_url();
    CHECK(!url.empty());
    auto channel = grpc::CreateChannel(url, grpc::InsecureChannelCredentials());
    m_stub       = mrc::protos::Architect::NewStub(channel);

    if (m_owns_progress_engine)
    {
        CHECK(m_cq);
        auto progress_engine = std::make_unique<rpc::ProgressEngine>(m_cq);
        m_progress_handler   = std::make_unique<rpc::PromiseHandler>();

        mrc::make_edge(*progress_engine, *m_progress_handler);

        m_progress_engine =
            runnable().launch_control().prepare_launcher(launch_options(), std::move(progress_engine))->ignition();
    }

    auto prepare_fn = [this](grpc::ClientContext* context) {
        CHECK(m_stub);
        return m_stub->PrepareAsyncEventStream(context, m_cq.get());
    };

    // event handler - optionally add concurrency here
    auto event_handler = std::make_unique<node::RxSink<event_t>>([this](event_t event) {
        do_handle_event(std::move(event));
    });

    // make stream and attach event handler
    m_stream = std::make_shared<stream_t::element_type>(prepare_fn, runnable());
    m_stream->attach_to(*event_handler);

    // ensure all downstream event handlers are constructed before constructing and starting the event handler
    m_connections_update_channel = std::make_unique<mrc::node::WritableEntrypoint<const protos::StateUpdate>>();
    m_connections_manager        = std::make_unique<client::ConnectionsManager>(*this, *m_connections_update_channel);

    // launch runnables
    m_event_handler =
        runnable().launch_control().prepare_launcher(launch_options(), std::move(event_handler))->ignition();

    // await initialization
    m_writer = m_stream->await_init();

    if (!m_writer)
    {
        forward_state(State::FailedToConnect);
        LOG(FATAL) << "unable to connect to control plane";
    }

    forward_state(State::Connected);
}

void Client::do_service_stop()
{
    m_writer->finish();
    m_writer.reset();
}

void Client::do_service_kill()
{
    m_writer->cancel();
    m_writer.reset();
}

void Client::do_service_await_live()
{
    if (m_owns_progress_engine)
    {
        m_progress_engine->await_live();
    }
    m_event_handler->await_live();
}

void Client::do_service_await_join()
{
    auto status = m_stream->await_fini();
    m_event_handler->await_join();
    m_connections_update_channel.reset();

    if (m_owns_progress_engine)
    {
        m_cq->Shutdown();
        m_progress_engine->await_join();
    }
}

void Client::do_handle_event(event_t&& event)
{
    switch (event.msg.event())
    {
        // handle a subset of events directly on the event handler

    case protos::EventType::Response: {
        auto event_tag = event.msg.tag();

        if (event_tag != 0)
        {
            // Lock to prevent multiple threads
            std::unique_lock<decltype(m_mutex)> lock(m_mutex);

            // Find the promise associated with the event tag
            auto promise = m_pending_events.extract(event_tag);

            // Unlock to allow other threads to continue as soon as possible
            lock.unlock();

            // Finally, set the value
            promise.mapped().set_value(std::move(event.msg));
        }
    }
    break;

    case protos::EventType::ServerStateUpdate: {
        protos::StateUpdate update;
        CHECK(event.msg.has_message() && event.msg.message().UnpackTo(&update));

        if (update.has_connections())
        {
            DCHECK(m_connections_update_channel);
            CHECK(m_connections_update_channel->await_write(std::move(update)) == channel::Status::success);
        }
        else
        {
            route_state_update(event.msg.tag(), std::move(update));
        }
    }
    break;

    default:
        LOG(FATAL) << "event channel not implemented";
    }
}

std::map<InstanceID, std::unique_ptr<client::Instance>> Client::register_ucx_addresses(
    std::vector<std::optional<ucx::UcxResources>>& ucx_resources)
{
    forward_state(State::RegisteringWorkers);
    auto instances = m_connections_manager->register_ucx_addresses(ucx_resources);
    forward_state(State::Operational);
    return instances;
}

void Client::forward_state(State state)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    CHECK(m_state < state);
    m_state = state;
}

void Client::route_state_update(std::uint64_t tag, protos::StateUpdate&& update)
{
    DCHECK(!m_connections_manager->instance_channels().empty());

    if (tag == 0)
    {
        DVLOG(10) << "broadcasting " << update.service_name() << " update to all instances";
        for (const auto& [id, instance] : m_connections_manager->instance_channels())
        {
            auto copy   = update;
            auto status = instance->await_write(std::move(copy));
            LOG_IF(WARNING, status != mrc::channel::Status::success)
                << "unable to route update for service: " << update.service_name();
        }
    }
    else
    {
        auto instance = m_connections_manager->instance_channels().find(tag);
        CHECK(instance != m_connections_manager->instance_channels().end());
        auto status = instance->second->await_write(std::move(update));
        LOG_IF(WARNING, status != mrc::channel::Status::success)
            << "unable to route update for service: " << update.service_name();
    }
}

// bool Client::has_subscription_service(const std::string& name) const
// {
//     std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//     return contains(m_subscription_services, name);
// }

const mrc::runnable::LaunchOptions& Client::launch_options() const
{
    return m_launch_options;
}

AsyncEventStatus Client::issue_event(const protos::EventType& event_type)
{
    protos::Event event;
    event.set_event(event_type);
    return this->write_event(std::move(event), false);
}

void Client::request_update()
{
    issue_event(protos::ClientEventRequestStateUpdate);
    // std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    // if (!m_update_in_progress && !m_update_requested)
    // {
    //     m_update_requested = true;
    //     issue_event(protos::ClientEventRequestStateUpdate);
    // }
}

AsyncEventStatus Client::write_event(protos::Event event, bool await_response)
{
    if (event.tag() != 0)
    {
        LOG(WARNING) << "event tag is set but this field should exclusively be used by the control plane client. "
                        "Clearing to avoid confusion";
        event.clear_tag();
    }

    AsyncEventStatus status;

    if (await_response)
    {
        // If we are supporting awaiting, create the promise now
        Promise<protos::Event> promise;

        // Set the future to the status
        status.set_future(promise.get_future());

        // Set the tag to the request ID to allow looking up the promise later
        event.set_tag(status.request_id());

        // Save the promise to the pending promises to be retrieved later
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        m_pending_events[status.request_id()] = std::move(promise);
    }

    // Finally, write the event
    m_writer->await_write(std::move(event));

    return status;
}
}  // namespace mrc::control_plane

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

#include "internal/control_plane/client.hpp"

#include "srf/channel/channel.hpp"
#include "srf/channel/status.hpp"
#include "srf/exceptions/runtime_error.hpp"
#include "srf/protos/architect.grpc.pb.h"
#include "srf/protos/architect.pb.h"

namespace srf::internal::control_plane {

Client::Client(runnable::Resources& runnable, std::shared_ptr<grpc::CompletionQueue> cq) :
  m_runnable(runnable),
  m_cq(std::move(cq)),
  m_owns_progress_engine(false)
{}

Client::Client(runnable::Resources& runnable) :
  m_runnable(runnable),
  m_cq(std::make_shared<grpc::CompletionQueue>()),
  m_owns_progress_engine(true)
{}

Client::~Client()
{
    Service::call_in_destructor();
}

void Client::do_service_start()
{
    auto url = m_runnable.system().options().architect_url();
    CHECK(!url.empty());
    auto channel = grpc::CreateChannel(url, grpc::InsecureChannelCredentials());
    m_stub       = srf::protos::Architect::NewStub(channel);

    if (m_owns_progress_engine)
    {
        CHECK(m_cq);
        auto progress_engine  = std::make_unique<rpc::ProgressEngine>(m_cq);
        auto progress_handler = std::make_unique<rpc::PromiseHandler>();

        srf::node::make_edge(*progress_engine, *progress_handler);

        m_progress_handler = m_runnable.launch_control().prepare_launcher(std::move(progress_handler))->ignition();
        m_progress_engine  = m_runnable.launch_control().prepare_launcher(std::move(progress_engine))->ignition();
    }

    auto prepare_fn = [this](grpc::ClientContext* context) {
        CHECK(m_stub);
        return m_stub->PrepareAsyncEventStream(context, m_cq.get());
    };

    // event handler - optionally add concurrency here
    auto event_handler =
        std::make_unique<node::RxSink<event_t>>([this](event_t event) { do_handle_event(std::move(event)); });

    // make stream and attach event handler
    m_stream = std::make_shared<stream_t::element_type>(prepare_fn, m_runnable);
    m_stream->attach_to(*event_handler);

    // launch runnables
    m_event_handler = m_runnable.launch_control().prepare_launcher(std::move(event_handler))->ignition();

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
        m_progress_handler->await_live();
    }
    m_event_handler->await_live();
}

void Client::do_service_await_join()
{
    auto status = m_stream->await_fini();
    m_event_handler->await_join();

    if (m_owns_progress_engine)
    {
        m_cq->Shutdown();
        m_progress_engine->await_join();
        m_progress_handler->await_join();
    }
}

void Client::do_handle_event(event_t&& event)
{
    switch (event.msg.event())
    {
        // handle a subset of events directly on the event handler

    case protos::EventType::Response: {
        auto* promise = reinterpret_cast<Promise<protos::Event>*>(event.msg.tag());
        if (promise != nullptr)
        {
            promise->set_value(std::move(event.msg));
        }
    }
    break;

    case protos::EventType::ServerUpdateSubscriptionService:
        route_subscription_service_update(std::move(event));
        break;

        // some events are routed to handlers on a given partition
        // e.g. pipeline updates are issued per partition and routed directly to the pipeline manager

    default:
        LOG(FATAL) << "event channel not implemented";
    }
}

void Client::register_ucx_addresses(std::vector<ucx::WorkerAddress> worker_addresses)
{
    forward_state(State::RegisteringWorkers);
    protos::RegisterWorkersRequest req;
    for (const auto& addr : worker_addresses)
    {
        req.add_ucx_worker_addresses(addr);
    }
    auto resp = await_unary<protos::RegisterWorkersResponse>(protos::ClientRegisterWorkers, std::move(req));

    m_machine_id = resp.machine_id();
    CHECK_EQ(resp.instance_ids_size(), worker_addresses.size());
    for (const auto& id : resp.instance_ids())
    {
        m_instance_ids.push_back(id);
    }

    DVLOG(10) << "control plane - machine_id: " << m_machine_id;
    forward_state(State::Operational);
}

client::SubscriptionService& Client::get_or_create_subscription_service(std::string name, std::set<std::string> roles)
{
    // no need to call the server if we have a locally registered subscription service
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        auto search = m_subscription_services.find(name);
        if (search != m_subscription_services.end())
        {
            // possible match - validate roles
            CHECK(search->second->roles() == roles);
            return *search->second;
        }
    }

    // issue unary rpc on the bidi stream
    protos::CreateSubscriptionServiceRequest req;
    req.set_service_name(name);
    for (const auto& role : roles)
    {
        req.add_roles(role);
    }
    auto resp = await_unary<protos::Ack>(protos::ClientUnaryCreateSubscriptionService, std::move(req));
    if (resp.status() != protos::Success)
    {
        LOG(ERROR) << resp.msg();
        throw srf::exceptions::SrfRuntimeError("failed to create subscription service");
    }
    DVLOG(10) << "subscribtion_service: " << name << " is live on the control plane server";

    // lock state - if not present, add subscriptions service specific update channel to the routing map
    // we had released the lock, so it's possible that multiple local paritions requested the same service
    // the first one here will register it
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    auto search = m_subscription_services.find(name);
    if (search == m_subscription_services.end())
    {
        m_subscription_services[name] = std::make_unique<client::SubscriptionService>(name, roles);
        return *m_subscription_services.at(name);
    }
    DCHECK(search->second->roles() == roles);
    return *search->second;
}

MachineID Client::machine_id() const
{
    return m_machine_id;
}

const std::vector<InstanceID>& Client::instance_ids() const
{
    return m_instance_ids;
}

void Client::forward_state(State state)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    CHECK(m_state < state);
    m_state = state;
}

void Client::route_subscription_service_update(event_t event)
{
    protos::SubscriptionServiceUpdate update;
    CHECK(event.msg.has_message() && event.msg.message().UnpackTo(&update));

    auto search = m_subscription_services.find(update.service_name());
    CHECK(search != m_subscription_services.end());

    auto state = search->second->await_write(std::move(update));
    LOG_IF(WARNING, state != srf::channel::Status::success)
        << "unable to route update for service: " << update.service_name();
}

}  // namespace srf::internal::control_plane

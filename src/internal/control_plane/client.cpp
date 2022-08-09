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

#include "internal/ucx/resources.hpp"
#include "internal/utils/contains.hpp"

#include "srf/channel/channel.hpp"
#include "srf/channel/status.hpp"
#include "srf/exceptions/runtime_error.hpp"
#include "srf/node/source_channel.hpp"
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
    m_launch_options.engine_factory_name = "main";
    m_launch_options.pe_count            = 1;
    m_launch_options.engines_per_pe      = 1;

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

        m_progress_handler =
            m_runnable.launch_control().prepare_launcher(launch_options(), std::move(progress_handler))->ignition();
        m_progress_engine =
            m_runnable.launch_control().prepare_launcher(launch_options(), std::move(progress_engine))->ignition();
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
    m_event_handler =
        m_runnable.launch_control().prepare_launcher(launch_options(), std::move(event_handler))->ignition();

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

    case protos::EventType::ServerStateUpdate:
        route_state_update(std::move(event));
        break;

    default:
        LOG(FATAL) << "event channel not implemented";
    }
}

std::map<InstanceID, std::unique_ptr<client::Instance>> Client::register_ucx_addresses(
    std::vector<std::optional<ucx::Resources>>& ucx_resources)
{
    forward_state(State::RegisteringWorkers);
    protos::RegisterWorkersRequest req;
    for (auto& ucx : ucx_resources)
    {
        DCHECK(ucx);
        req.add_ucx_worker_addresses(ucx->worker().address());
    }
    auto resp = await_unary<protos::RegisterWorkersResponse>(protos::ClientUnaryRegisterWorkers, std::move(req));

    m_machine_id = resp->machine_id();
    CHECK_EQ(resp->instance_ids_size(), ucx_resources.size());
    std::map<InstanceID, std::unique_ptr<client::Instance>> instances;
    for (int i = 0; i < resp->instance_ids_size(); i++)
    {
        auto id = resp->instance_ids().at(i);
        m_instance_ids.push_back(id);
        m_update_channels[id] = std::make_unique<update_channel_t>();
        instances[id] = std::make_unique<client::Instance>(*this, id, *ucx_resources.at(i), *m_update_channels.at(id));
    }

    // issue activate event - connection events from the server will
    issue_event(protos::ClientEventActivateStream, std::move(*resp));

    DVLOG(10) << "control plane - machine_id: " << m_machine_id;
    forward_state(State::Operational);
    return instances;
}

void Client::drop_instance(const InstanceID& instance_id)
{
    protos::TaggedInstance msg;
    msg.set_instance_id(instance_id);
    msg.set_tag(m_machine_id);
    await_unary<protos::Ack>(protos::ClientUnaryDropWorker, std::move(msg));
    // DCHECK(contains(m_update_channels, instance_id));
    m_update_channels.erase(instance_id);
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
    if (!resp)
    {
        LOG(ERROR) << "failed to create subscription service: " << resp.error().message();
        throw resp.error();
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

void Client::route_state_update(event_t event)
{
    protos::StateUpdate update;
    DCHECK(event.msg.event() == protos::ServerStateUpdate);
    CHECK(event.msg.has_message() && event.msg.message().UnpackTo(&update));

    if (event.msg.tag() == 0)
    {
        DVLOG(10) << "broadcasting " << update.service_name() << " update to all instances";
        for (const auto& [id, instance] : m_update_channels)
        {
            auto copy   = update;
            auto status = instance->await_write(std::move(copy));
            LOG_IF(WARNING, status != srf::channel::Status::success)
                << "unable to route update for service: " << update.service_name();
        }
    }
    else
    {
        auto instance = m_update_channels.find(event.msg.tag());
        CHECK(instance != m_update_channels.end());
        auto status = instance->second->await_write(std::move(update));
        LOG_IF(WARNING, status != srf::channel::Status::success)
            << "unable to route update for service: " << update.service_name();
    }
}

bool Client::has_subscription_service(const std::string& name) const
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return contains(m_subscription_services, name);
}

const runnable::LaunchOptions& Client::launch_options() const
{
    return m_launch_options;
}

}  // namespace srf::internal::control_plane

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

#include "internal/control_plane/client/connections_manager.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/ucx/resources.hpp"
#include "internal/utils/contains.hpp"

#include "srf/channel/channel.hpp"
#include "srf/channel/status.hpp"
#include "srf/exceptions/runtime_error.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/protos/architect.grpc.pb.h"
#include "srf/protos/architect.pb.h"

namespace srf::internal::control_plane {

Client::Client(resources::PartitionResourceBase& base, std::shared_ptr<grpc::CompletionQueue> cq) :
  resources::PartitionResourceBase(base),
  m_cq(std::move(cq)),
  m_owns_progress_engine(false)
{}

Client::Client(resources::PartitionResourceBase& base) :
  resources::PartitionResourceBase(base),
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
    m_stub       = srf::protos::Architect::NewStub(channel);

    if (m_owns_progress_engine)
    {
        CHECK(m_cq);
        auto progress_engine  = std::make_unique<rpc::ProgressEngine>(m_cq);
        auto progress_handler = std::make_unique<rpc::PromiseHandler>();

        srf::node::make_edge(*progress_engine, *progress_handler);

        m_progress_handler =
            runnable().launch_control().prepare_launcher(launch_options(), std::move(progress_handler))->ignition();
        m_progress_engine =
            runnable().launch_control().prepare_launcher(launch_options(), std::move(progress_engine))->ignition();
    }

    auto prepare_fn = [this](grpc::ClientContext* context) {
        CHECK(m_stub);
        return m_stub->PrepareAsyncEventStream(context, m_cq.get());
    };

    // event handler - optionally add concurrency here
    auto event_handler =
        std::make_unique<node::RxSink<event_t>>([this](event_t event) { do_handle_event(std::move(event)); });

    // make stream and attach event handler
    m_stream = std::make_shared<stream_t::element_type>(prepare_fn, runnable());
    m_stream->attach_to(*event_handler);

    // ensure all downstream event handlers are constructed before constructing and starting the event handler
    m_update_channel      = std::make_unique<srf::node::SourceChannelWriteable<const protos::StateUpdate>>();
    m_connections_manager = std::make_unique<client::ConnectionsManager>(*this, *m_update_channel);

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
        m_progress_handler->await_live();
    }
    m_event_handler->await_live();
}

void Client::do_service_await_join()
{
    auto status = m_stream->await_fini();
    m_event_handler->await_join();
    m_update_channel.reset();

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

    case protos::EventType::ServerStateUpdate: {
        protos::StateUpdate update;
        CHECK(event.msg.has_message() && event.msg.message().UnpackTo(&update));
        DCHECK(m_update_channel);
        CHECK(m_update_channel->await_write(std::move(update)) == channel::Status::success);
    }
    break;

    default:
        LOG(FATAL) << "event channel not implemented";
    }
}

std::map<InstanceID, std::unique_ptr<client::Instance>> Client::register_ucx_addresses(
    std::vector<std::optional<ucx::Resources>>& ucx_resources)
{
    forward_state(State::RegisteringWorkers);
    auto instances = m_connections_manager->register_ucx_addresses(ucx_resources);
    forward_state(State::Operational);
    return instances;
}

// client::SubscriptionService& Client::get_or_create_subscription_service(std::string name, std::set<std::string>
// roles)
// {
//     // no need to call the server if we have a locally registered subscription service
//     CHECK(runnable().main().caller_on_same_thread());

//     {
//         std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//         auto search = m_subscription_services.find(name);
//         if (search != m_subscription_services.end())
//         {
//             // possible match - validate roles
//             CHECK(search->second->roles() == roles);
//             return *search->second;
//         }
//     }

//     // issue unary rpc on the bidi stream
//     protos::CreateSubscriptionServiceRequest req;
//     req.set_service_name(name);
//     for (const auto& role : roles)
//     {
//         req.add_roles(role);
//     }
//     auto resp = await_unary<protos::Ack>(protos::ClientUnaryCreateSubscriptionService, std::move(req));
//     if (!resp)
//     {
//         LOG(ERROR) << "failed to create subscription service: " << resp.error().message();
//         throw resp.error();
//     }
//     DVLOG(10) << "subscribtion_service: " << name << " is live on the control plane server";

//     // lock state - if not present, add subscriptions service specific update channel to the routing map
//     // we had released the lock, so it's possible that multiple local paritions requested the same service
//     // the first one here will register it
//     std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//     auto search = m_subscription_services.find(name);
//     if (search == m_subscription_services.end())
//     {
//         m_subscription_services[name] = std::make_unique<client::SubscriptionService>(name, roles);
//         return *m_subscription_services.at(name);
//     }
//     DCHECK(search->second->roles() == roles);
//     return *search->second;
// }

// MachineID Client::machine_id() const
// {
//     return m_machine_id;
// }

// const std::vector<InstanceID>& Client::instance_ids() const
// {
//     std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//     return m_instance_ids;
// }

void Client::forward_state(State state)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    CHECK(m_state < state);
    m_state = state;
}

// void Client::route_state_update_event(const event_t& event)
// {
//     protos::StateUpdate update;
//     DCHECK(event.msg.event() == protos::ServerStateUpdate);
//     CHECK(event.msg.has_message() && event.msg.message().UnpackTo(&update));

//     // check for connections
//     if (update.has_connections())
//     {
//         // m_connections_manager.update(std::move(update));
//     }
//     else
//     {
//         // protect m_update_channels
//         std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//         route_state_update(event.msg.tag(), std::move(update));
//     }
// }

// void Client::route_state_update(std::uint64_t tag, protos::StateUpdate&& update)
// {
//     if (tag == 0)
//     {
//         DVLOG(10) << "broadcasting " << update.service_name() << " update to all instances";
//         for (const auto& [id, instance] : m_update_channels)
//         {
//             auto copy   = update;
//             auto status = instance->await_write(std::move(copy));
//             LOG_IF(WARNING, status != srf::channel::Status::success)
//                 << "unable to route update for service: " << update.service_name();
//         }
//     }
//     else
//     {
//         auto instance = m_update_channels.find(tag);
//         CHECK(instance != m_update_channels.end());
//         auto status = instance->second->await_write(std::move(update));
//         LOG_IF(WARNING, status != srf::channel::Status::success)
//             << "unable to route update for service: " << update.service_name();
//     }
// }

// bool Client::has_subscription_service(const std::string& name) const
// {
//     std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//     return contains(m_subscription_services, name);
// }

const runnable::LaunchOptions& Client::launch_options() const
{
    return m_launch_options;
}

void Client::issue_event(const protos::EventType& event_type)
{
    protos::Event event;
    event.set_event(event_type);
    m_writer->await_write(std::move(event));
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

}  // namespace srf::internal::control_plane

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

#include "internal/control_plane/server/client_instance.hpp"
#include "internal/control_plane/server/subscription_service.hpp"
#include "internal/grpc/server.hpp"
#include "internal/grpc/server_streaming.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/service.hpp"

#include "srf/node/queue.hpp"
#include "srf/protos/architect.grpc.pb.h"
#include "srf/protos/architect.pb.h"
#include "srf/runnable/runner.hpp"

#include <boost/fiber/recursive_mutex.hpp>

#include <map>
#include <memory>
#include <string>

namespace srf::internal::control_plane {

class Server : public Service
{
  public:
    using stream_t      = std::shared_ptr<rpc::ServerStream<srf::protos::Event, srf::protos::Event>>;
    using writer_t      = std::shared_ptr<rpc::StreamWriter<srf::protos::Event>>;
    using event_t       = stream_t::element_type::IncomingData;
    using instance_t    = std::shared_ptr<server::ClientInstance>;
    using stream_id_t   = std::size_t;
    using instance_id_t = std::size_t;

    Server(runnable::Resources& runnable);

  private:
    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;

    void do_accept_stream(rxcpp::subscriber<stream_t>& s);
    void do_handle_event(event_t&& event);

    // top-level event handlers - these methods lock internal state
    void unary_register_workers(event_t& event);
    void unary_create_subscription_service(event_t& event);
    // todo(ryan) - convert to unary service with an ack response
    void register_subscription_service(event_t& event);
    void unary_drop_from_subscription_service(event_t& event);
    void drop_stream(writer_t writer);

    static void unary_ack(event_t& event, protos::ErrorCode type, std::string msg = "");

    // convenience methods - these method do not lock internal state
    std::shared_ptr<server::ClientInstance> get_instance(const instance_id_t& instance_id) const;
    bool validate_instance_id(const instance_id_t& instance_id, const event_t& event) const;
    bool has_subscription_service(const std::string& name) const;

    // srf resources
    runnable::Resources& m_runnable;

    // grpc
    rpc::Server m_server;
    std::shared_ptr<srf::protos::Architect::AsyncService> m_service;

    // connection info
    std::map<stream_id_t, stream_t> m_streams;
    std::map<instance_id_t, std::shared_ptr<server::ClientInstance>> m_instances;
    std::multimap<stream_id_t, instance_id_t> m_instances_by_stream;
    std::set<std::string> m_ucx_worker_addresses;

    // subscription services
    std::map<std::string, std::unique_ptr<SubscriptionService>> m_subscription_services;

    // operators / queues
    std::unique_ptr<srf::node::Queue<event_t>> m_queue;

    // runners
    std::unique_ptr<srf::runnable::Runner> m_stream_acceptor;
    std::unique_ptr<srf::runnable::Runner> m_event_handler;

    mutable boost::fibers::mutex m_mutex;
};

}  // namespace srf::internal::control_plane

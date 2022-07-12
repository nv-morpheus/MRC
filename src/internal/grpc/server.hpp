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

#include "internal/runnable/resources.hpp"
#include "internal/service.hpp"

#include "srf/runnable/runner.hpp"

#include <grpcpp/completion_queue.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>

namespace srf::internal::rpc::server {

class Server : public Service
{
  public:
    Server(runnable::Resources& runnable);
    ~Server() override;

    void register_service(std::shared_ptr<grpc::Service> service);

    std::shared_ptr<grpc::ServerCompletionQueue> get_cq() const;

    runnable::Resources& runnable();

  private:
    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;

    grpc::ServerBuilder m_builder;
    runnable::Resources& m_runnable;
    std::vector<std::shared_ptr<grpc::Service>> m_services;
    std::shared_ptr<grpc::ServerCompletionQueue> m_cq;
    std::unique_ptr<grpc::Server> m_server;
    std::unique_ptr<srf::runnable::Runner> m_progress_engine;
    std::unique_ptr<srf::runnable::Runner> m_event_hander;
};

}  // namespace srf::internal::rpc::server

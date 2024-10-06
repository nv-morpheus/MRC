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

#pragma once

#include "internal/service.hpp"

#include <grpcpp/grpcpp.h>

#include <memory>
#include <vector>

namespace grpc {
class Service;
}  // namespace grpc
namespace mrc::runnable {
class RunnableResources;
}  // namespace mrc::runnable
namespace mrc::runnable {
class Runner;
}  // namespace mrc::runnable

namespace mrc::rpc {
class PromiseHandler;
}  // namespace mrc::rpc

namespace mrc::rpc {

class Server : public Service
{
  public:
    Server(runnable::RunnableResources& runnable);
    ~Server() override;

    void register_service(std::shared_ptr<grpc::Service> service);

    std::shared_ptr<grpc::ServerCompletionQueue> get_cq() const;

    runnable::RunnableResources& runnable();

  private:
    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;

    grpc::ServerBuilder m_builder;
    runnable::RunnableResources& m_runnable;
    std::vector<std::shared_ptr<grpc::Service>> m_services;
    std::shared_ptr<grpc::ServerCompletionQueue> m_cq;
    std::unique_ptr<grpc::Server> m_server;
    std::unique_ptr<mrc::runnable::Runner> m_progress_engine;
    std::unique_ptr<mrc::rpc::PromiseHandler> m_event_hander;
};

}  // namespace mrc::rpc

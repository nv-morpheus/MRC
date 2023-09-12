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

#include "internal/grpc/server.hpp"

#include "internal/grpc/progress_engine.hpp"
#include "internal/runnable/runnable_resources.hpp"

#include "mrc/edge/edge_builder.hpp"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runnable/runner.hpp"

#include <memory>
#include <utility>

namespace mrc::rpc {

Server::Server(runnable::RunnableResources& runnable) : m_runnable(runnable)
{
    m_cq = m_builder.AddCompletionQueue();
    m_builder.AddListeningPort("0.0.0.0:13337", grpc::InsecureServerCredentials());
}

Server::~Server()
{
    Service::call_in_destructor();
}

void Server::do_service_start()
{
    m_server = m_builder.BuildAndStart();

    auto progress_engine = std::make_unique<ProgressEngine>(m_cq);
    m_event_hander       = std::make_unique<PromiseHandler>();
    mrc::make_edge(*progress_engine, *m_event_hander);

    m_progress_engine = m_runnable.launch_control().prepare_launcher(std::move(progress_engine))->ignition();
}

void Server::do_service_stop()
{
    service_kill();
}

void Server::do_service_kill()
{
    if (m_server)
    {
        m_server->Shutdown();
        m_cq->Shutdown();
    }
}

void Server::do_service_await_live()
{
    if (m_progress_engine)
    {
        m_progress_engine->await_live();
    }
}

void Server::do_service_await_join()
{
    if (m_progress_engine)
    {
        m_progress_engine->await_join();
    }
}

runnable::RunnableResources& Server::runnable()
{
    return m_runnable;
}
std::shared_ptr<grpc::ServerCompletionQueue> Server::get_cq() const
{
    return m_cq;
}
void Server::register_service(std::shared_ptr<grpc::Service> service)
{
    m_builder.RegisterService(service.get());
    m_services.push_back(service);
}
}  // namespace mrc::rpc

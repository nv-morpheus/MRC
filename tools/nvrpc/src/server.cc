/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvrpc/server.h>

#include <nvrpc/interfaces.h>

#include <glog/logging.h>
#include <grpcpp/security/server_credentials.h>  // for InsecureServerCreden...

#include <csignal>
#include <ostream>  // for logging ostream<<

namespace {
std::function<void(int)> shutdown_handler;
void signal_handler(int signal)
{
    shutdown_handler(signal);
}
}  // namespace

namespace nvrpc {

Server::Server(std::string server_address) : m_ServerAddress(server_address), m_Running(false)
{
    VLOG(1) << "gRPC will listening on: " << m_ServerAddress;
    m_Builder.AddListeningPort(m_ServerAddress, ::grpc::InsecureServerCredentials());
}

::grpc::ServerBuilder& Server::Builder()
{
    LOG_IF(FATAL, m_Running) << "Unable to access Builder after the Server is running.";
    return m_Builder;
}

void Server::Run()
{
    Run(std::chrono::milliseconds(1000), [] {});
}

void Server::Run(std::chrono::milliseconds timeout, std::function<void()> control_fn)
{
    AsyncStart();
    while (m_Running)
    {
        {
            std::unique_lock<std::mutex> lock(m_Mutex);
            if (m_Condition.wait_for(lock, timeout, [this] { return !m_Running; }))
            {
                // if not running
                m_Condition.notify_all();
                DLOG(INFO) << "Server::Run exitting";
                return;
            }
            else
            {
                // if running
                // DLOG(INFO) << "Server::Run executing user lambda";
                control_fn();
            }
        }
    }
}

void Server::AsyncStart()
{
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        CHECK_EQ(m_Running, false) << "Server is already running";
        m_Server = m_Builder.BuildAndStart();

        shutdown_handler = [this](int signal) {
            LOG(INFO) << "Trapped Signal: " << signal;
            Shutdown();
        };
        std::signal(SIGINT, signal_handler);

        for (int i = 0; i < m_Executors.size(); i++)
        {
            m_Executors[i]->Run();
        }
        m_Running = true;
    }
    m_Condition.notify_all();
    VLOG(1) << "grpc server and event loop initialized and accepting connections";
}

void Server::Shutdown()
{
    VLOG(1) << "Shutdown Requested";
    CHECK(m_Server);
    {
        std::lock_guard<std::mutex> lock(m_Mutex);
        m_Server->Shutdown();
        for (auto& executor : m_Executors)
        {
            executor->Shutdown();
        }
        m_Running = false;
    }
    m_Condition.notify_all();
}

bool Server::Running()
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    return m_Running;
}

}  // namespace nvrpc

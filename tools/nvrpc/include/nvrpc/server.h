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

#pragma once

#ifndef NVIS_SERVER_H_
#define NVIS_SERVER_H_

#include <nvrpc/interfaces.h>  // for IExecutor, IService
#include <nvrpc/service.h>

#include <grpcpp/grpcpp.h>  // for Server, ServerBuilder

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>  // for function
#include <memory>      // for unique_ptr, allocator
#include <mutex>
#include <stdexcept>  // for runtime_error
#include <string>
#include <vector>

namespace nvrpc {

using std::chrono::milliseconds;

class Server
{
  public:
    Server(std::string server_address);

    Server() : Server("0.0.0.0:50051") {}

    template <class ServiceType>
    AsyncService<typename ServiceType::AsyncService>* RegisterAsyncService();

    IExecutor* RegisterExecutor(IExecutor* executor)
    {
        m_Executors.emplace_back(executor);
        executor->Initialize(m_Builder);
        return executor;
    }

    void Run();
    void Run(milliseconds timeout, std::function<void()> control_fn);
    void AsyncStart();
    void Shutdown();

    bool Running();

    ::grpc::ServerBuilder& Builder();

  private:
    bool m_Running;
    std::mutex m_Mutex;
    std::condition_variable m_Condition;
    std::string m_ServerAddress;
    ::grpc::ServerBuilder m_Builder;
    std::unique_ptr<::grpc::Server> m_Server;
    std::vector<std::unique_ptr<IService>> m_Services;
    std::vector<std::unique_ptr<IExecutor>> m_Executors;
};

template <class ServiceType>
AsyncService<typename ServiceType::AsyncService>* Server::RegisterAsyncService()
{
    if (m_Running)
    {
        throw std::runtime_error("Error: cannot register service on a running server");
    }
    auto service = new AsyncService<typename ServiceType::AsyncService>;
    auto base    = static_cast<IService*>(service);
    m_Services.emplace_back(base);
    service->Initialize(m_Builder);
    return service;
}

}  // namespace nvrpc

#endif  // NVIS_SERVER_H_

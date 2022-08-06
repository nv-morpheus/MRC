/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "test_resources.h"

#include "testing.grpc.pb.h"
#include "testing.pb.h"  // IWYU pragma: keep

#include <nvrpc/executor.h>
#include <nvrpc/server.h>

#include <memory>   // for unique_ptr, make_shared, make_unique
#include <utility>  // for move

namespace nvrpc {
namespace testing {

template <typename Context>
std::unique_ptr<Server> BuildServer();

template <typename T, typename ExecutorType = Executor>
std::unique_ptr<Server> BuildStreamingServer()
{
    auto server        = std::make_unique<Server>("0.0.0.0:13377");
    auto resources     = std::make_shared<TestResources>(3);
    auto executor      = server->RegisterExecutor(new ExecutorType(1));
    auto service       = server->RegisterAsyncService<TestService>();
    auto rpc_streaming = service->RegisterRPC<T>(&TestService::AsyncService::RequestStreaming);
    executor->RegisterContexts(rpc_streaming, resources, 10);
    return std::move(server);
}

template <typename UnaryContext, typename StreamingContext, typename ExecutorType = Executor>
std::unique_ptr<Server> BuildServer()
{
    auto server        = std::make_unique<Server>("0.0.0.0:13377");
    auto resources     = std::make_shared<TestResources>(3);
    auto executor      = server->RegisterExecutor(new ExecutorType(1));
    auto service       = server->RegisterAsyncService<TestService>();
    auto rpc_unary     = service->RegisterRPC<UnaryContext>(&TestService::AsyncService::RequestUnary);
    auto rpc_streaming = service->RegisterRPC<StreamingContext>(&TestService::AsyncService::RequestStreaming);
    executor->RegisterContexts(rpc_unary, resources, 10);
    executor->RegisterContexts(rpc_streaming, resources, 10);
    return std::move(server);
}

}  // namespace testing
}  // namespace nvrpc

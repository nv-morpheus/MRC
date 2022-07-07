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

#include "common.hpp"

#include "internal/grpc/server/server.hpp"
#include "internal/grpc/server/stream.hpp"
#include "internal/resources/manager.hpp"

#include "srf/codable/codable_protocol.hpp"
#include "srf/codable/fundamental_types.hpp"
#include "srf/protos/architect.grpc.pb.h"
#include "srf/protos/architect.pb.h"

#include <glog/logging.h>
#include <grpcpp/server_context.h>
#include <gtest/gtest.h>

using namespace srf;
using namespace srf::codable;

class TestRPC : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_resources = std::make_unique<internal::resources::Manager>(
            internal::system::SystemProvider(make_system([](Options& options) {
                // todo(#114) - propose: remove this option entirely
                // options.architect_url("localhost:13337");
                options.topology().user_cpuset("0-3");
                options.topology().restrict_gpus(true);
                options.placement().resources_strategy(PlacementResources::Dedicated);
            })));
    }

    void TearDown() override
    {
        m_resources.reset();
    }

    std::unique_ptr<internal::resources::Manager> m_resources;
};

TEST_F(TestRPC, ServerLifeCycle)
{
    internal::rpc::server::Server server(m_resources->partition(0).runnable());

    server.service_start();
    server.service_await_live();
    server.service_stop();
    server.service_await_join();

    // auto service = std::make_shared<srf::protos::Architect::AsyncService>();
    // server.register_service(service);
}

class TestStream : internal::rpc::server::StreamContext<srf::protos::Event, srf::protos::Event>
{
    using base_t = internal::rpc::server::StreamContext<srf::protos::Event, srf::protos::Event>;

    void handler(srf::protos::Event&& event, const Stream& stream) final {}

    void on_initialized() final {}
    void on_write_done() final {}
    void on_write_fail() final {}

  public:
    using base_t::base_t;
};

TEST_F(TestRPC, StreamLifeCycle)
{
    internal::rpc::server::Server server(m_resources->partition(0).runnable());

    auto service = std::make_shared<srf::protos::Architect::AsyncService>();
    server.register_service(service);

    auto cq           = server.get_cq();
    auto service_init = [service, cq](grpc::ServerContext* context,
                                      grpc::ServerAsyncReaderWriter<::srf::protos::Event, ::srf::protos::Event>* stream,
                                      void* tag) {
        service->RequestEventStream(context, stream, cq.get(), cq.get(), tag);
    };

    auto stream = std::make_shared<TestStream>(service_init, m_resources->partition(0).runnable());

    server.service_start();
    server.service_await_live();

    server.service_stop();
    server.service_await_join();

    stream.reset();
}

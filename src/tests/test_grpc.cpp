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

#include "internal/grpc/client/stream.hpp"
#include "internal/grpc/server/server.hpp"
#include "internal/grpc/server/stream.hpp"
#include "internal/resources/manager.hpp"

#include "srf/codable/codable_protocol.hpp"
#include "srf/codable/fundamental_types.hpp"
#include "srf/node/forward.hpp"
#include "srf/protos/architect.grpc.pb.h"
#include "srf/protos/architect.pb.h"
#include "srf/protos/test.grpc.pb.h"
#include "srf/protos/test.pb.h"

#include <glog/logging.h>
#include <grpcpp/client_context.h>
#include <grpcpp/server_context.h>
#include <gtest/gtest.h>

#include <set>
#include <thread>

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

        m_server = std::make_unique<internal::rpc::server::Server>(m_resources->partition(0).runnable());

        m_channel = grpc::CreateChannel("localhost:13337", grpc::InsecureChannelCredentials());
        m_stub    = srf::testing::TestService::NewStub(m_channel);
    }

    void TearDown() override
    {
        m_stub.reset();
        m_channel.reset();
        m_server.reset();
        m_resources.reset();
    }

    std::unique_ptr<internal::resources::Manager> m_resources;
    std::shared_ptr<grpc::Channel> m_channel;
    std::shared_ptr<srf::testing::TestService::Stub> m_stub;
    std::unique_ptr<internal::rpc::server::Server> m_server;
};

TEST_F(TestRPC, ServerLifeCycle)
{
    m_server->service_start();
    m_server->service_await_live();
    m_server->service_stop();
    m_server->service_await_join();

    // auto service = std::make_shared<srf::protos::Architect::AsyncService>();
    // server.register_service(service);
}

class PingPongServerContext : internal::rpc::server::StreamContext<srf::testing::Input, srf::testing::Output>
{
    using base_t = internal::rpc::server::StreamContext<srf::testing::Input, srf::testing::Output>;

    void on_request(srf::testing::Input&& input, Stream& stream) final
    {
        srf::testing::Output output;
        if (input.batch_id() == 42)
        {
            return;
        }
        output.set_batch_id(input.batch_id());
        stream.await_write(std::move(output));
    }

    void on_initialized() final
    {
        LOG(INFO) << "server steaming context initialized";
    }
    void on_write_done() final {}
    void on_write_fail() final {}

  public:
    using base_t::base_t;
};

TEST_F(TestRPC, ServerStreamShutdownBeforeStreamInit)
{
    auto service = std::make_shared<srf::testing::TestService::AsyncService>();
    m_server->register_service(service);

    auto cq           = m_server->get_cq();
    auto service_init = [service, cq](grpc::ServerContext* context,
                                      grpc::ServerAsyncReaderWriter<srf::testing::Output, srf::testing::Input>* stream,
                                      void* tag) {
        service->RequestStreaming(context, stream, cq.get(), cq.get(), tag);
    };

    m_server->service_start();
    m_server->service_await_live();

    auto stream = std::make_shared<PingPongServerContext>(service_init, m_resources->partition(0).runnable());

    m_server->service_stop();
    m_server->service_await_join();

    stream.reset();
}

template <typename WriteT>
struct Callbacks
{
    std::function<void(bool)> on_reader_complete;
    std::function<void(bool)> on_fini;
    std::function<void(std::optional<internal::rpc::Stream<WriteT>>)> on_init;
};

class PingPongClientContext : public internal::rpc::client::StreamContext<srf::testing::Input, srf::testing::Output>
{
  public:
    PingPongClientContext(std::unique_ptr<Callbacks<srf::testing::Input>> callbacks,
                          prepare_fn_t prepare_fn,
                          internal::runnable::Resources& runnable) :
      internal::rpc::client::StreamContext<srf::testing::Input, srf::testing::Output>(prepare_fn, runnable),
      m_callbacks(std::move(callbacks))
    {
        CHECK(m_callbacks);
    }

  private:
    void on_read(srf::testing::Output&& request, const internal::rpc::Stream<srf::testing::Input>& stream) final {}

    void on_reader_complete(bool ok) final
    {
        m_callbacks->on_reader_complete(ok);
    }
    void on_init(std::optional<internal::rpc::Stream<srf::testing::Input>> stream) final
    {
        m_callbacks->on_init(std::move(stream));
    }
    void on_fini(bool ok) final
    {
        m_callbacks->on_fini(ok);
    }

    const std::unique_ptr<Callbacks<srf::testing::Input>> m_callbacks;
};

TEST_F(TestRPC, PingPong)
{
    GTEST_SKIP() << "currently hangs";

    auto service = std::make_shared<srf::testing::TestService::AsyncService>();
    m_server->register_service(service);

    auto cq           = m_server->get_cq();
    auto service_init = [service, cq](grpc::ServerContext* context,
                                      grpc::ServerAsyncReaderWriter<srf::testing::Output, srf::testing::Input>* stream,
                                      void* tag) {
        service->RequestStreaming(context, stream, cq.get(), cq.get(), tag);
    };

    m_server->service_start();
    m_server->service_await_live();

    auto server_stream_ctx =
        std::make_shared<PingPongServerContext>(service_init, m_resources->partition(0).runnable());

    auto prepare_fn = [this, cq](grpc::ClientContext* context) {
        return m_stub->PrepareAsyncStreaming(context, cq.get());
    };

    auto callbacks = std::make_unique<Callbacks<srf::testing::Input>>();

    callbacks->on_init = [](std::optional<internal::rpc::Stream<srf::testing::Input>> stream) { EXPECT_TRUE(stream); };

    auto client_stream_ctx =
        std::make_shared<PingPongClientContext>(std::move(callbacks), prepare_fn, m_resources->partition(0).runnable());

    client_stream_ctx->service_start();
    client_stream_ctx->service_await_live();

    std::this_thread::sleep_for(std::chrono::seconds(1));

    m_server->service_stop();
    m_server->service_await_join();

    client_stream_ctx->service_await_join();
    server_stream_ctx.reset();
}

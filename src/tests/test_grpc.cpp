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

#include "internal/grpc/client_streaming.hpp"
#include "internal/grpc/server.hpp"
#include "internal/grpc/server_streaming.hpp"
#include "internal/resources/manager.hpp"

#include "srf/channel/status.hpp"
#include "srf/codable/codable_protocol.hpp"
#include "srf/codable/fundamental_types.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/generic_sink.hpp"
#include "srf/node/sink_properties.hpp"
#include "srf/protos/architect.grpc.pb.h"
#include "srf/protos/architect.pb.h"
#include "srf/protos/test.grpc.pb.h"
#include "srf/protos/test.pb.h"

#include <glog/logging.h>
#include <grpcpp/client_context.h>
#include <grpcpp/completion_queue.h>
#include <grpcpp/server_context.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <optional>
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
                options.topology().user_cpuset("0-8");
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

using stream_server_t = internal::rpc::ServerStream<srf::testing::Input, srf::testing::Output>;
using stream_client_t = internal::rpc::ClientStream<srf::testing::Input, srf::testing::Output>;

TEST_F(TestRPC, Alternative)
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

    auto stream = std::make_shared<stream_server_t>(service_init, m_resources->partition(0).runnable());

    auto f_writer = m_resources->partition(0).runnable().main().enqueue([stream] { return stream->await_init(); });
    m_resources->partition(0).runnable().main().enqueue([] {}).get();

    m_server->service_stop();
    m_server->service_await_join();

    auto writer = f_writer.get();
    EXPECT_TRUE(writer == nullptr);

    auto status = stream->await_fini();
    EXPECT_FALSE(status.ok());
}

class ServerHandler : public srf::node::GenericSink<typename stream_server_t::IncomingData>
{
    void on_data(typename stream_server_t::IncomingData&& data) final
    {
        if (data.ok && !m_done_or_cancelled)
        {
            srf::testing::Output response;
            if (data.msg.batch_id() == 42)
            {
                m_done_or_cancelled = true;
                data.stream->finish();
                return;
            }
            if (data.msg.batch_id() == 420)
            {
                m_done_or_cancelled = true;
                data.stream->cancel();
                return;
            }
            response.set_batch_id(data.msg.batch_id());
            data.stream->await_write(std::move(response));
        }
    }

    bool m_done_or_cancelled{false};
};

// this test will simply bring up the grpc server and await on a server stream which will not get initialized
// because no client connects and the server shuts down
TEST_F(TestRPC, StreamingServerWithHandler)
{
    auto service = std::make_shared<srf::testing::TestService::AsyncService>();
    m_server->register_service(service);

    auto cq           = m_server->get_cq();
    auto service_init = [service, cq](grpc::ServerContext* context,
                                      grpc::ServerAsyncReaderWriter<srf::testing::Output, srf::testing::Input>* stream,
                                      void* tag) {
        service->RequestStreaming(context, stream, cq.get(), cq.get(), tag);
    };

    auto stream  = std::make_shared<stream_server_t>(service_init, m_resources->partition(0).runnable());
    auto handler = std::make_unique<ServerHandler>();
    handler->enable_persistence();
    stream->attach_to(*handler);

    auto handler_runner =
        m_resources->partition(0).runnable().launch_control().prepare_launcher(std::move(handler))->ignition();
    handler_runner->await_live();

    m_server->service_start();
    m_server->service_await_live();

    auto f_writer = m_resources->partition(0).runnable().main().enqueue([stream] { return stream->await_init(); });
    m_resources->partition(0).runnable().main().enqueue([] {}).get();  // this is a fence

    m_server->service_stop();
    m_server->service_await_join();

    auto writer = f_writer.get();
    EXPECT_TRUE(writer == nullptr);

    auto status = stream->await_fini();
    EXPECT_FALSE(status.ok());

    handler_runner->stop();
    handler_runner->await_join();
}

// basic client server ping pong
// client issues a request, server returns it, client awaits the return by reading from the handler, then repeats
// this tests the basic lifecycle of a normally behaving client/server stream.
TEST_F(TestRPC, StreamingPingPong)
{
    auto service = std::make_shared<srf::testing::TestService::AsyncService>();
    m_server->register_service(service);

    auto cq           = m_server->get_cq();
    auto service_init = [service, cq](grpc::ServerContext* context,
                                      grpc::ServerAsyncReaderWriter<srf::testing::Output, srf::testing::Input>* stream,
                                      void* tag) {
        service->RequestStreaming(context, stream, cq.get(), cq.get(), tag);
    };

    auto stream  = std::make_shared<stream_server_t>(service_init, m_resources->partition(0).runnable());
    auto handler = std::make_unique<ServerHandler>();
    handler->enable_persistence();
    stream->attach_to(*handler);

    auto handler_runner =
        m_resources->partition(0).runnable().launch_control().prepare_launcher(std::move(handler))->ignition();
    handler_runner->await_live();

    m_server->service_start();
    m_server->service_await_live();

    m_resources->partition(0).runnable().main().enqueue([stream] { return stream->await_init(); });
    m_resources->partition(0).runnable().main().enqueue([] {}).get();  // this is a fence

    // put client here
    auto prepare_fn = [this, cq](grpc::ClientContext* context) {
        return m_stub->PrepareAsyncStreaming(context, cq.get());
    };

    auto client = std::make_shared<stream_client_t>(prepare_fn, m_resources->partition(0).runnable());
    srf::node::SinkChannelReadable<typename stream_client_t::IncomingData> client_handler;
    client->attach_to(client_handler);

    auto client_writer = client->await_init();
    ASSERT_TRUE(client_writer);
    for (int i = 0; i < 10; i++)
    {
        VLOG(1) << "sending request " << i;
        srf::testing::Input request;
        request.set_batch_id(i);
        client_writer->await_write(std::move(request));

        typename stream_client_t::IncomingData response;
        VLOG(1) << "awaiting response " << i;
        client_handler.egress().await_read(response);
        VLOG(1) << "got response " << i;

        EXPECT_EQ(response.response.batch_id(), i);
    }

    client_writer->finish();
    client_writer.reset();  // this should issue writes done to the server and begin shutdown

    auto client_status = client->await_fini();
    EXPECT_TRUE(client_status.ok());

    // ensure client is done before shutting down server

    m_server->service_stop();
    m_server->service_await_join();

    auto status = stream->await_fini();
    EXPECT_TRUE(status.ok());

    handler_runner->stop();
    handler_runner->await_join();
}

TEST_F(TestRPC, StreamingPingPongEarlyServerFinish)
{
    auto service = std::make_shared<srf::testing::TestService::AsyncService>();
    m_server->register_service(service);

    auto cq           = m_server->get_cq();
    auto service_init = [service, cq](grpc::ServerContext* context,
                                      grpc::ServerAsyncReaderWriter<srf::testing::Output, srf::testing::Input>* stream,
                                      void* tag) {
        service->RequestStreaming(context, stream, cq.get(), cq.get(), tag);
    };

    auto stream  = std::make_shared<stream_server_t>(service_init, m_resources->partition(0).runnable());
    auto handler = std::make_unique<ServerHandler>();
    handler->enable_persistence();
    stream->attach_to(*handler);

    auto handler_runner =
        m_resources->partition(0).runnable().launch_control().prepare_launcher(std::move(handler))->ignition();
    handler_runner->await_live();

    m_server->service_start();
    m_server->service_await_live();

    m_resources->partition(0).runnable().main().enqueue([stream] { return stream->await_init(); });
    m_resources->partition(0).runnable().main().enqueue([] {}).get();  // this is a fence

    // put client here
    auto prepare_fn = [this, cq](grpc::ClientContext* context) {
        return m_stub->PrepareAsyncStreaming(context, cq.get());
    };

    auto client = std::make_shared<stream_client_t>(prepare_fn, m_resources->partition(0).runnable());
    srf::node::SinkChannelReadable<typename stream_client_t::IncomingData> client_handler;
    client->attach_to(client_handler);

    auto client_writer = client->await_init();
    ASSERT_TRUE(client_writer);
    for (int i = 40; i < 50; i++)
    {
        VLOG(1) << "sending request " << i;
        srf::testing::Input request;
        request.set_batch_id(i);
        auto status = client_writer->await_write(std::move(request));
        if (i <= 42)
        {
            EXPECT_EQ(status, srf::channel::Status::success);
        }
        else
        {
            EXPECT_EQ(status, srf::channel::Status::closed);
        }

        if (status == srf::channel::Status::success)
        {
            typename stream_client_t::IncomingData response;
            VLOG(1) << "awaiting response " << i;
            auto status = client_handler.egress().await_read(response);

            if (status == srf::channel::Status::success)
            {
                VLOG(1) << "got response " << i;
                EXPECT_EQ(response.response.batch_id(), i);
            }
        }
    }

    client_writer->finish();
    client_writer.reset();  // this should issue writes done to the server and begin shutdown

    // for grpc, either the client or the server can finish/cancel the stream at anytime
    // the server is the ultimate decider of the truth, so if it finishes early, even though the client might have more
    // to say, ultimiately, if the server returns OK, the client will accept the result and finish here with OK.
    auto client_status = client->await_fini();
    EXPECT_TRUE(client_status.ok());

    // ensure client is done before shutting down server

    m_server->service_stop();
    m_server->service_await_join();

    // the server which finished with status OK, returns OK here.
    auto status = stream->await_fini();
    EXPECT_TRUE(status.ok());

    handler_runner->stop();
    handler_runner->await_join();
}

TEST_F(TestRPC, StreamingPingPongEarlyServerCancel)
{
    auto service = std::make_shared<srf::testing::TestService::AsyncService>();
    m_server->register_service(service);

    auto cq           = m_server->get_cq();
    auto service_init = [service, cq](grpc::ServerContext* context,
                                      grpc::ServerAsyncReaderWriter<srf::testing::Output, srf::testing::Input>* stream,
                                      void* tag) {
        service->RequestStreaming(context, stream, cq.get(), cq.get(), tag);
    };

    auto stream  = std::make_shared<stream_server_t>(service_init, m_resources->partition(0).runnable());
    auto handler = std::make_unique<ServerHandler>();
    handler->enable_persistence();
    stream->attach_to(*handler);

    auto handler_runner =
        m_resources->partition(0).runnable().launch_control().prepare_launcher(std::move(handler))->ignition();
    handler_runner->await_live();

    m_server->service_start();
    m_server->service_await_live();

    m_resources->partition(0).runnable().main().enqueue([stream] { return stream->await_init(); });
    m_resources->partition(0).runnable().main().enqueue([] {}).get();  // this is a fence

    // put client here
    auto prepare_fn = [this, cq](grpc::ClientContext* context) {
        return m_stub->PrepareAsyncStreaming(context, cq.get());
    };

    auto client = std::make_shared<stream_client_t>(prepare_fn, m_resources->partition(0).runnable());
    srf::node::SinkChannelReadable<typename stream_client_t::IncomingData> client_handler;
    client->attach_to(client_handler);

    auto client_writer = client->await_init();
    ASSERT_TRUE(client_writer);
    for (int i = 400; i < 500; i += 10)
    {
        VLOG(1) << "sending request " << i;
        srf::testing::Input request;
        request.set_batch_id(i);
        auto status = client_writer->await_write(std::move(request));
        if (i <= 420)
        {
            EXPECT_EQ(status, srf::channel::Status::success);
        }
        else
        {
            EXPECT_EQ(status, srf::channel::Status::closed);
        }

        if (status == srf::channel::Status::success)
        {
            typename stream_client_t::IncomingData response;
            VLOG(1) << "awaiting response " << i;
            auto status = client_handler.egress().await_read(response);

            if (status == srf::channel::Status::success)
            {
                VLOG(1) << "got response " << i;
                EXPECT_EQ(response.response.batch_id(), i);
            }
        }
    }

    client_writer->finish();
    client_writer.reset();  // this should issue writes done to the server and begin shutdown

    // for grpc, either the client or the server can finish/cancel the stream at anytime; however,
    // to get an OK result here, I (ryan) am of the understanding that the client needs to have issued
    // a WritesDone, then the server is allowed to Finish with status OK.
    // In this case, the server finishes with status OK, but the client gets back CANCELLED
    auto client_status = client->await_fini();
    EXPECT_FALSE(client_status.ok());

    // ensure client is done before shutting down server

    m_server->service_stop();
    m_server->service_await_join();

    // the server which finished with status OK, returns OK here.
    auto status = stream->await_fini();
    EXPECT_FALSE(status.ok());

    handler_runner->stop();
    handler_runner->await_join();
}

TEST_F(TestRPC, StreamingPingPongClientEarlyTermination)
{
    auto service = std::make_shared<srf::testing::TestService::AsyncService>();
    m_server->register_service(service);

    auto cq           = m_server->get_cq();
    auto service_init = [service, cq](grpc::ServerContext* context,
                                      grpc::ServerAsyncReaderWriter<srf::testing::Output, srf::testing::Input>* stream,
                                      void* tag) {
        service->RequestStreaming(context, stream, cq.get(), cq.get(), tag);
    };

    auto stream  = std::make_shared<stream_server_t>(service_init, m_resources->partition(0).runnable());
    auto handler = std::make_unique<ServerHandler>();
    handler->enable_persistence();
    stream->attach_to(*handler);

    auto handler_runner =
        m_resources->partition(0).runnable().launch_control().prepare_launcher(std::move(handler))->ignition();
    handler_runner->await_live();

    m_server->service_start();
    m_server->service_await_live();

    m_resources->partition(0).runnable().main().enqueue([stream] { return stream->await_init(); });
    m_resources->partition(0).runnable().main().enqueue([] {}).get();  // this is a fence

    // put client here
    auto prepare_fn = [this, cq](grpc::ClientContext* context) {
        return m_stub->PrepareAsyncStreaming(context, cq.get());
    };

    auto client = std::make_shared<stream_client_t>(prepare_fn, m_resources->partition(0).runnable());
    srf::node::SinkChannelReadable<typename stream_client_t::IncomingData> client_handler;
    client->attach_to(client_handler);

    auto client_writer = client->await_init();
    ASSERT_TRUE(client_writer);
    for (int i = 0; i < 10; i++)
    {
        VLOG(1) << "sending request " << i;
        srf::testing::Input request;
        request.set_batch_id(i);
        client_writer->await_write(std::move(request));

        typename stream_client_t::IncomingData response;
        VLOG(1) << "awaiting response " << i;
        client_handler.egress().await_read(response);
        VLOG(1) << "got response " << i;

        EXPECT_EQ(response.response.batch_id(), i);
    }

    client_writer->cancel();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    client_writer.reset();  // this should issue writes done to the server and begin shutdown

    auto client_status = client->await_fini();
    EXPECT_FALSE(client_status.ok());

    // the server doesn't really know if the client cancelled or just issued a WritesDone
    // if the server tries to write to the channel, it will realize that it can not and return
    // a CANCELLED here, however, if it treats the cancel as a WritesDone and does not issues
    // any responses to the client, it will finish with OK.
    // we might be able to construct a test that issues a cancel with the server in a yielding state, then have the
    // server issue a write after the server side reader is done. in that scenario, we should then see a CANCELLED
    // this example finishes with OK.
    auto status = stream->await_fini();
    EXPECT_TRUE(status.ok());

    m_server->service_stop();
    m_server->service_await_join();

    handler_runner->stop();
    handler_runner->await_join();
}

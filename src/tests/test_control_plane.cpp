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

#include "internal/control_plane/server.hpp"
#include "internal/grpc/client_streaming.hpp"
#include "internal/grpc/server.hpp"
#include "internal/grpc/server_streaming.hpp"
#include "internal/resources/manager.hpp"

#include "srf/node/sink_properties.hpp"
#include "srf/protos/architect.grpc.pb.h"
#include "srf/protos/architect.pb.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <optional>
#include <set>
#include <thread>

using namespace srf;

class TestControlPlane : public ::testing::Test
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

        m_channel = grpc::CreateChannel("localhost:13337", grpc::InsecureChannelCredentials());
        m_stub    = srf::protos::Architect::NewStub(m_channel);
    }

    void TearDown() override
    {
        m_stub.reset();
        m_channel.reset();
        m_resources.reset();
    }

    std::unique_ptr<internal::resources::Manager> m_resources;
    std::shared_ptr<grpc::Channel> m_channel;
    std::shared_ptr<srf::protos::Architect::Stub> m_stub;
};

TEST_F(TestControlPlane, LifeCycle)
{
    auto server = std::make_unique<internal::control_plane::Server>(m_resources->partition(0).runnable());

    server->service_start();
    server->service_await_live();

    // inspect server

    server->service_stop();
    server->service_await_join();
}

TEST_F(TestControlPlane, SingleClientConnectDisconnect)
{
    auto server = std::make_unique<internal::control_plane::Server>(m_resources->partition(0).runnable());

    server->service_start();
    server->service_await_live();

    // convert the following code into control_plane::Client
    // put client here
    // auto prepare_fn = [this, cq](grpc::ClientContext* context) {
    //     return m_stub->PrepareAsyncEventStream(context, cq.get());
    // };

    // auto client = std::make_shared<stream_client_t>(prepare_fn, m_resources->partition(0).runnable());
    // srf::node::SinkChannelReadable<typename stream_client_t::IncomingData> client_handler;
    // client->attach_to(client_handler);

    // auto client_writer = client->await_init();

    server->service_stop();
    server->service_await_join();
}

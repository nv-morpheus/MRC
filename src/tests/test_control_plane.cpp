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

#include "internal/control_plane/client.hpp"
#include "internal/control_plane/resources.hpp"
#include "internal/control_plane/server.hpp"
#include "internal/data_plane/resources.hpp"
#include "internal/grpc/client_streaming.hpp"
#include "internal/grpc/server.hpp"
#include "internal/grpc/server_streaming.hpp"
#include "internal/network/resources.hpp"
#include "internal/resources/manager.hpp"

#include "srf/node/sink_properties.hpp"
#include "srf/options/placement.hpp"
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

static auto make_resources(std::function<void(Options& options)> options_lambda = [](Options& options) {})
{
    return std::make_unique<internal::resources::Manager>(
        internal::system::SystemProvider(make_system([&](Options& options) {
            options.topology().user_cpuset("0-8");
            options.topology().restrict_gpus(true);
            options.placement().resources_strategy(PlacementResources::Dedicated);
            options.placement().cpu_strategy(PlacementStrategy::PerMachine);
            options_lambda(options);
        })));
}

class TestControlPlane : public ::testing::Test
{
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestControlPlane, LifeCycle)
{
    auto sr     = make_resources();
    auto server = std::make_unique<internal::control_plane::Server>(sr->partition(0).runnable());

    server->service_start();
    server->service_await_live();

    // inspect server

    server->service_stop();
    server->service_await_join();
}

TEST_F(TestControlPlane, SingleClientConnectDisconnect)
{
    auto sr     = make_resources();
    auto server = std::make_unique<internal::control_plane::Server>(sr->partition(0).runnable());

    server->service_start();
    server->service_await_live();

    auto cr = make_resources([](Options& options) { options.architect_url("localhost:13337"); });

    // the total number of partition is system dependent
    auto expected_partitions = cr->system().partitions().flattened().size();
    EXPECT_EQ(cr->partition(0).network()->control_plane().client().instance_ids().size(), expected_partitions);

    // destroying the resources should gracefully shutdown the data plane and the control plane.
    cr.reset();

    // auto client = std::make_unique<internal::control_plane::Client>(m_resources->partition(0).runnable());
    // EXPECT_EQ(client->state(), state_t::Disconnected);
    // client->service_start();
    // client->service_await_live();
    // EXPECT_EQ(client->state(), state_t::Connected);

    // client->register_ucx_addresses({m_resources->partition(0).network()->data_plane().ucx_address()});
    // EXPECT_EQ(client->instance_ids().size(), 1);
    // EXPECT_EQ(client->state(), state_t::Operational);

    // EXPECT_FALSE(client->has_subscription_service("port_knox"));
    // client->get_or_create_subscription_service("port_knox", {"publisher", "subscriber"});
    // EXPECT_TRUE(client->has_subscription_service("port_knox"));

    // client->service_stop();
    // client->service_await_join();

    server->service_stop();
    server->service_await_join();
}

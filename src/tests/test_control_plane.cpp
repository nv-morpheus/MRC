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
#include "internal/control_plane/client/connections_manager.hpp"
#include "internal/control_plane/resources.hpp"
#include "internal/control_plane/server.hpp"
#include "internal/data_plane/resources.hpp"
#include "internal/grpc/client_streaming.hpp"
#include "internal/grpc/server.hpp"
#include "internal/grpc/server_streaming.hpp"
#include "internal/network/resources.hpp"
#include "internal/pubsub/publisher_manager.hpp"
#include "internal/pubsub/subscriber_manager.hpp"
#include "internal/resources/manager.hpp"

#include "srf/codable/fundamental_types.hpp"  // IWYU pragma: keep
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
            options.topology().user_cpuset("0-3");
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
    EXPECT_EQ(cr->partition(0).network()->control_plane().client().connections().instance_ids().size(),
              expected_partitions);

    // destroying the resources should gracefully shutdown the data plane and the control plane.
    cr.reset();

    server->service_stop();
    server->service_await_join();
}

TEST_F(TestControlPlane, DoubleClientConnectExchangeDisconnect)
{
    auto sr     = make_resources();
    auto server = std::make_unique<internal::control_plane::Server>(sr->partition(0).runnable());

    server->service_start();
    server->service_await_live();

    auto client_1 = make_resources([](Options& options) {
        options.topology().user_cpuset("0-3");
        options.topology().restrict_gpus(true);
        options.architect_url("localhost:13337");
    });

    auto client_2 = make_resources([](Options& options) {
        options.topology().user_cpuset("4-7");
        options.topology().restrict_gpus(true);
        options.architect_url("localhost:13337");
    });

    // the total number of partition is system dependent
    auto expected_partitions_1 = client_1->system().partitions().flattened().size();
    EXPECT_EQ(client_1->partition(0).network()->control_plane().client().connections().instance_ids().size(),
              expected_partitions_1);

    auto expected_partitions_2 = client_2->system().partitions().flattened().size();
    EXPECT_EQ(client_2->partition(0).network()->control_plane().client().connections().instance_ids().size(),
              expected_partitions_2);

    auto f1 = client_1->partition(0).network()->control_plane().client().connections().update_future();
    auto f2 = client_2->partition(0).network()->control_plane().client().connections().update_future();

    client_1->partition(0).network()->control_plane().client().request_update();

    f1.get();
    f2.get();

    client_1->partition(0)
        .runnable()
        .main()
        .enqueue([&] {
            auto worker_count =
                client_1->partition(0).network()->control_plane().client().connections().worker_addresses().size();
            EXPECT_EQ(worker_count, expected_partitions_1 + expected_partitions_2);
        })
        .get();

    // destroying the resources should gracefully shutdown the data plane and the control plane.
    client_1.reset();
    client_2.reset();

    server->service_stop();
    server->service_await_join();
}

TEST_F(TestControlPlane, DoubleClientPubSub)
{
    auto sr     = make_resources();
    auto server = std::make_unique<internal::control_plane::Server>(sr->partition(0).runnable());

    server->service_start();
    server->service_await_live();

    auto client_1 = make_resources([](Options& options) {
        options.topology().user_cpuset("0-3");
        options.topology().restrict_gpus(true);
        options.architect_url("localhost:13337");
    });

    auto client_2 = make_resources([](Options& options) {
        options.topology().user_cpuset("4-7");
        options.topology().restrict_gpus(true);
        options.architect_url("localhost:13337");
    });

    // the total number of partition is system dependent
    auto expected_partitions_1 = client_1->system().partitions().flattened().size();
    EXPECT_EQ(client_1->partition(0).network()->control_plane().client().connections().instance_ids().size(),
              expected_partitions_1);

    auto expected_partitions_2 = client_2->system().partitions().flattened().size();
    EXPECT_EQ(client_2->partition(0).network()->control_plane().client().connections().instance_ids().size(),
              expected_partitions_2);

    auto f1 = client_1->partition(0).network()->control_plane().client().connections().update_future();
    auto f2 = client_2->partition(0).network()->control_plane().client().connections().update_future();

    client_1->partition(0).network()->control_plane().client().request_update();

    f1.get();
    f2.get();

    client_1->partition(0)
        .runnable()
        .main()
        .enqueue([&] {
            auto worker_count =
                client_1->partition(0).network()->control_plane().client().connections().worker_addresses().size();
            EXPECT_EQ(worker_count, expected_partitions_1 + expected_partitions_2);
        })
        .get();

    LOG(INFO) << "MAKE PUBLISHER";

    auto publisher = internal::pubsub::make_publisher<int>(
        "my_int", internal::pubsub::PublisherType::RoundRobin, client_1->partition(0));

    LOG(INFO) << "MAKE SUBSCRIBER";
    auto subscriber = internal::pubsub::make_subscriber<int>("my_int", client_2->partition(0));

    client_1->partition(0).network()->control_plane().client().request_update();

    publisher->await_write(42);
    publisher->await_write(15);

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    LOG(INFO) << "AFTER SLEEP 1 - publisher should have 1 subscriber";
    // client-side: publisher manager should have 1 tagged instance in it write list
    // server-side: publisher member list: 1, subscriber member list: 1, subscriber subscribe_to list: 1

    LOG(INFO) << "[START] DELETE SUBSCRIBER";
    subscriber.reset();
    LOG(INFO) << "[FINISH] DELETE SUBSCRIBER";

    client_1->partition(0).network()->control_plane().client().request_update();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    LOG(INFO) << "AFTER SLEEP 2 - publisher should have 0 subscribers";

    LOG(INFO) << "[START] DELETE PUBLISHER";
    publisher.reset();
    LOG(INFO) << "[FINISH] DELETE PUBLISHER";

    client_1->partition(0).network()->control_plane().client().request_update();

    // destroying the resources should gracefully shutdown the data plane and the control plane.
    client_1.reset();
    client_2.reset();

    server->service_stop();
    server->service_await_join();
}

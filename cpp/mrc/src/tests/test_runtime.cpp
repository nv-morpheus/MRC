/**
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

#include "common.hpp"

#include "internal/control_plane/client.hpp"
#include "internal/control_plane/client/connections_manager.hpp"
#include "internal/control_plane/client/instance.hpp"
#include "internal/control_plane/server.hpp"
#include "internal/network/network_resources.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/runnable/runnable_resources.hpp"
#include "internal/runtime/partition_runtime.hpp"
#include "internal/runtime/runtime.hpp"
#include "internal/system/partitions.hpp"
#include "internal/system/system.hpp"
#include "internal/system/system_provider.hpp"

#include "mrc/codable/fundamental_types.hpp"  // IWYU pragma: keep
#include "mrc/core/task_queue.hpp"
#include "mrc/memory/literals.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/placement.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/pubsub/api.hpp"
#include "mrc/pubsub/forward.hpp"
#include "mrc/pubsub/publisher.hpp"
#include "mrc/pubsub/subscriber.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <gtest/internal/gtest-internal.h>

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <thread>
#include <utility>
#include <vector>

using namespace mrc::runtime;
using namespace mrc::memory::literals;

namespace mrc::runtime {

static auto make_resources(std::function<void(Options& options)> options_lambda = [](Options& options) {})
{
    auto resources = std::make_unique<resources::SystemResources>(
        system::SystemProvider(make_system([&](Options& options) {
            options.topology().user_cpuset("0-3");
            options.topology().restrict_gpus(true);
            options.placement().resources_strategy(PlacementResources::Dedicated);
            options.placement().cpu_strategy(PlacementStrategy::PerMachine);
            options_lambda(options);
        })));

    return resources;
}

class TestRuntime : public ::testing::Test
{
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestRuntime, LifeCycle)
{
    auto resources = make_resources();

    auto runtime = std::make_unique<Runtime>(*resources);

    runtime->service_start();
    runtime->service_await_live();

    runtime->service_stop();
    runtime->service_await_join();
}
}  // namespace mrc::runtime

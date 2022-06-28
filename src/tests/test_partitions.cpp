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

#include "internal/system/gpu_info.hpp"
#include "internal/system/host_partition.hpp"
#include "internal/system/partitions.hpp"
#include "internal/system/topology.hpp"

#include "srf/core/bitmap.hpp"
#include "srf/options/options.hpp"
#include "srf/options/placement.hpp"
#include "srf/options/topology.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace srf;
using namespace internal;
using namespace system;

// iwyu is getting confused between std::uint32_t and boost::uint32_t
// IWYU pragma: no_include <boost/cstdint.hpp>

// todo(ryan) - further parameterize these unit tests to read in a `GetParam()`.json file with the expected host and
// device partitions for device paritions, validate both the cpu_set and the pcie device address are correct

static void add_engine_factory_services(Options& options)
{
    EngineFactoryOptions group;
    group.engine_type   = runnable::EngineType::Fiber;
    group.allow_overlap = false;
    group.reusable      = true;
    group.cpu_count     = 1;
    options.engine_factories().set_engine_factory_options("services", std::move(group));
}

static void add_engine_factory_dedicated_threads(Options& options)
{
    EngineFactoryOptions group;
    group.engine_type   = runnable::EngineType::Thread;
    group.allow_overlap = false;
    group.reusable      = false;
    group.cpu_count     = 2;
    options.engine_factories().set_engine_factory_options("dedicated_threads", std::move(group));
}

static void add_engine_factory_dedicated_fibers(Options& options)
{
    EngineFactoryOptions group;
    group.engine_type   = runnable::EngineType::Fiber;
    group.allow_overlap = false;
    group.reusable      = false;
    group.cpu_count     = 2;
    options.engine_factories().set_engine_factory_options("dedicated_fibers", std::move(group));
}

static void add_engine_factory_shared_fibers(Options& options)
{
    EngineFactoryOptions group;
    group.engine_type   = runnable::EngineType::Fiber;
    group.allow_overlap = true;
    group.reusable      = true;
    group.cpu_count     = 2;
    options.engine_factories().set_engine_factory_options("shared_fibers", std::move(group));
}

static void add_engine_factory_shared_threads_x1(Options& options)
{
    EngineFactoryOptions group;
    group.engine_type   = runnable::EngineType::Thread;
    group.allow_overlap = true;
    group.reusable      = true;
    group.cpu_count     = 1;
    options.engine_factories().set_engine_factory_options("shared_threads_x1", std::move(group));
}

static void add_engine_factory_shared_threads_x2(Options& options)
{
    EngineFactoryOptions group;
    group.engine_type   = runnable::EngineType::Thread;
    group.allow_overlap = true;
    group.reusable      = true;
    group.cpu_count     = 2;
    options.engine_factories().set_engine_factory_options("shared_threads_x2", std::move(group));
}

class TestPartitions : public testing::TestWithParam<const char*>
{
  protected:
    static std::shared_ptr<Options> make_options(std::function<void(Options&)> updater = nullptr)
    {
        auto options = std::make_shared<Options>();
        if (updater)
        {
            updater(*options);
        }

        return options;
    }

    static std::unique_ptr<Partitions> make_partitions(std::shared_ptr<Options> options)
    {
        std::string root_path;
        std::stringstream path;
        auto* root_env = std::getenv("SRF_TEST_INTERNAL_DATA_PATH");
        if (root_env != nullptr)
        {
            root_path = std::string(root_env);
        }
        else
        {
            root_path = ".";
        }
        path << root_path << "/data/" << std::string(GetParam()) << ".bin";
        VLOG(10) << "root_data_path: " << path.str();
        // todo(backlog) - assert the the fixture fie exists
        auto topo_proto = Topology::deserialize_from_file(path.str());
        auto topology   = Topology::Create(options->topology(), topo_proto);
        return std::make_unique<Partitions>(*topology, *options);
    }
};

TEST_P(TestPartitions, Scenario1)
{
    auto options =
        make_options([](Options& options) { options.placement().resources_strategy(PlacementResources::Dedicated); });
    auto partitions = make_partitions(options);

    EXPECT_EQ(partitions->device_partitions().size(), 4);
    EXPECT_EQ(partitions->host_partitions().size(), 4);

    // todo(ryan) - coded loop based on json structure of fixture
    EXPECT_EQ(partitions->device_partitions().at(0).host().cpu_set().str(), "0-15,64-79");
    EXPECT_EQ(partitions->device_partitions().at(0).pcie_bus_id(), "00000000:C2:00.0");
    EXPECT_EQ(partitions->device_partitions().at(1).host().cpu_set().str(), "16-31,80-95");
    EXPECT_EQ(partitions->device_partitions().at(1).pcie_bus_id(), "00000000:81:00.0");
    EXPECT_EQ(partitions->device_partitions().at(2).host().cpu_set().str(), "32-47,96-111");
    EXPECT_EQ(partitions->device_partitions().at(2).pcie_bus_id(), "00000000:47:00.0");
    EXPECT_EQ(partitions->device_partitions().at(3).host().cpu_set().str(), "48-63,112-127");
    EXPECT_EQ(partitions->device_partitions().at(3).pcie_bus_id(), "00000000:01:00.0");
}

TEST_P(TestPartitions, Scenario2)
{
    auto options    = make_options([](Options& options) {
        options.topology().ignore_dgx_display(false);
        options.placement().resources_strategy(PlacementResources::Dedicated);
    });
    auto partitions = make_partitions(options);

    // todo(ryan) - coded loop based on json structure of fixture
    EXPECT_EQ(partitions->device_partitions().size(), 5);
    EXPECT_EQ(partitions->host_partitions().size(), 5);
    EXPECT_EQ(partitions->device_partitions().at(0).host().cpu_set().str(), "0-11,64-75");
    EXPECT_EQ(partitions->device_partitions().at(0).pcie_bus_id(), "00000000:C1:00.0");
    EXPECT_EQ(partitions->device_partitions().at(1).host().cpu_set().str(), "13-24,77-88");
    EXPECT_EQ(partitions->device_partitions().at(1).pcie_bus_id(), "00000000:C2:00.0");
    EXPECT_EQ(partitions->device_partitions().at(2).host().cpu_set().str(), "26-37,90-101");
    EXPECT_EQ(partitions->device_partitions().at(2).pcie_bus_id(), "00000000:81:00.0");
    EXPECT_EQ(partitions->device_partitions().at(3).host().cpu_set().str(), "39-50,103-114");
    EXPECT_EQ(partitions->device_partitions().at(3).pcie_bus_id(), "00000000:47:00.0");
    EXPECT_EQ(partitions->device_partitions().at(4).host().cpu_set().str(), "52-63,116-127");
    EXPECT_EQ(partitions->device_partitions().at(4).pcie_bus_id(), "00000000:01:00.0");
}

TEST_P(TestPartitions, Scenario3)
{
    auto options    = make_options([](Options& options) {
        options.topology().ignore_dgx_display(false);
        options.placement().cpu_strategy(PlacementStrategy::PerNumaNode);
        options.placement().resources_strategy(PlacementResources::Dedicated);
    });
    auto partitions = make_partitions(options);

    // todo(ryan) - coded loop based on json structure of fixture
    EXPECT_EQ(partitions->device_partitions().size(), 5);
    EXPECT_EQ(partitions->host_partitions().size(), 5);
    EXPECT_EQ(partitions->device_partitions().at(0).host().cpu_set().str(), "0-11,64-75");
    EXPECT_EQ(partitions->device_partitions().at(0).pcie_bus_id(), "00000000:C1:00.0");
    EXPECT_EQ(partitions->device_partitions().at(1).host().cpu_set().str(), "13-24,77-88");
    EXPECT_EQ(partitions->device_partitions().at(1).pcie_bus_id(), "00000000:C2:00.0");
    EXPECT_EQ(partitions->device_partitions().at(2).host().cpu_set().str(), "26-37,90-101");
    EXPECT_EQ(partitions->device_partitions().at(2).pcie_bus_id(), "00000000:81:00.0");
    EXPECT_EQ(partitions->device_partitions().at(3).host().cpu_set().str(), "39-50,103-114");
    EXPECT_EQ(partitions->device_partitions().at(3).pcie_bus_id(), "00000000:47:00.0");
    EXPECT_EQ(partitions->device_partitions().at(4).host().cpu_set().str(), "52-63,116-127");
    EXPECT_EQ(partitions->device_partitions().at(4).pcie_bus_id(), "00000000:01:00.0");
    EXPECT_TRUE(partitions->cpu_strategy() == PlacementStrategy::PerMachine)
        << "requested PerNuma should fall back to PerMachine if the topology is asymmetric";
}

TEST_P(TestPartitions, Scenario4)
{
    auto options    = make_options([](Options& options) {
        options.topology().user_cpuset("0-15");
        options.topology().restrict_gpus(true);
        options.topology().ignore_dgx_display(false);
        options.placement().cpu_strategy(PlacementStrategy::PerNumaNode);
        options.placement().resources_strategy(PlacementResources::Dedicated);
    });
    auto partitions = make_partitions(options);

    EXPECT_EQ(partitions->device_partitions().size(), 2);
    EXPECT_EQ(partitions->host_partitions().size(), 2);
    EXPECT_EQ(partitions->device_partitions().at(0).host().cpu_set().str(), "0-7");
    EXPECT_EQ(partitions->device_partitions().at(0).pcie_bus_id(), "00000000:C1:00.0");
    EXPECT_EQ(partitions->device_partitions().at(1).host().cpu_set().str(), "8-15");
    EXPECT_EQ(partitions->device_partitions().at(1).pcie_bus_id(), "00000000:C2:00.0");
    EXPECT_TRUE(partitions->cpu_strategy() == PlacementStrategy::PerNumaNode);
}

TEST_P(TestPartitions, Scenario5)
{
    auto options =
        make_options([](Options& options) { options.placement().resources_strategy(PlacementResources::Shared); });
    auto partitions = make_partitions(options);

    EXPECT_EQ(partitions->device_partitions().size(), 4);
    EXPECT_EQ(partitions->host_partitions().size(), 1);
    EXPECT_EQ(partitions->device_partitions().at(0).host().cpu_set().str(), "0-127");
}

TEST_P(TestPartitions, Scenario6)
{
    auto options    = make_options([](Options& options) {
        options.placement().cpu_strategy(PlacementStrategy::PerNumaNode);
        options.placement().resources_strategy(PlacementResources::Shared);
    });
    auto partitions = make_partitions(options);

    // todo(ryan) - coded loop based on json structure of fixture
    EXPECT_EQ(partitions->device_partitions().size(), 4);
    EXPECT_EQ(partitions->host_partitions().size(), 4);
    EXPECT_EQ(partitions->device_partitions().at(0).host().cpu_set().str(), "0-15,64-79");
    EXPECT_EQ(partitions->device_partitions().at(0).pcie_bus_id(), "00000000:C2:00.0");
    EXPECT_EQ(partitions->device_partitions().at(1).host().cpu_set().str(), "16-31,80-95");
    EXPECT_EQ(partitions->device_partitions().at(1).pcie_bus_id(), "00000000:81:00.0");
    EXPECT_EQ(partitions->device_partitions().at(2).host().cpu_set().str(), "32-47,96-111");
    EXPECT_EQ(partitions->device_partitions().at(2).pcie_bus_id(), "00000000:47:00.0");
    EXPECT_EQ(partitions->device_partitions().at(3).host().cpu_set().str(), "48-63,112-127");
    EXPECT_EQ(partitions->device_partitions().at(3).pcie_bus_id(), "00000000:01:00.0");
}

TEST_P(TestPartitions, Scenario7)
{
    auto options    = make_options([](Options& options) {
        options.topology().ignore_dgx_display(false);
        options.placement().cpu_strategy(PlacementStrategy::PerNumaNode);
        options.placement().resources_strategy(PlacementResources::Shared);
    });
    auto partitions = make_partitions(options);

    EXPECT_EQ(partitions->device_partitions().size(), 5);
    EXPECT_EQ(partitions->host_partitions().size(), 1);
    EXPECT_EQ(partitions->device_partitions().at(0).host().cpu_set().str(), "0-127");
}

TEST_P(TestPartitions, Scenario8)
{
    auto options    = make_options([](Options& options) {
        options.topology().user_cpuset("0-15");
        options.topology().restrict_gpus(true);
        options.topology().ignore_dgx_display(false);
        options.placement().cpu_strategy(PlacementStrategy::PerNumaNode);
        options.placement().resources_strategy(PlacementResources::Shared);
    });
    auto partitions = make_partitions(options);

    EXPECT_EQ(partitions->device_partitions().size(), 2);
    EXPECT_EQ(partitions->host_partitions().size(), 1);
    EXPECT_EQ(partitions->device_partitions().at(0).host().cpu_set().str(), "0-15");
    EXPECT_EQ(partitions->device_partitions().at(1).host().cpu_set().str(), "0-15");
    EXPECT_EQ(partitions->device_partitions().at(0).pcie_bus_id(), "00000000:C1:00.0");
    EXPECT_EQ(partitions->device_partitions().at(1).pcie_bus_id(), "00000000:C2:00.0");
    EXPECT_TRUE(partitions->cpu_strategy() == PlacementStrategy::PerNumaNode);
}

// note: this needs special consideration when generalizing
TEST_P(TestPartitions, SingleCore1GPU)
{
    auto options = make_options([](Options& options) {
        options.topology().user_cpuset("0");
        options.topology().restrict_gpus(true);
        options.placement().cpu_strategy(PlacementStrategy::PerMachine);
        options.placement().resources_strategy(PlacementResources::Shared);
    });

    auto partitions = make_partitions(options);

    EXPECT_EQ(partitions->host_partitions().size(), 1);
    EXPECT_EQ(partitions->device_partitions().size(), 1);
}

// note: this needs special consideration when generalizing
TEST_P(TestPartitions, SingleCore4GPU)
{
    auto options = make_options([](Options& options) {
        options.topology().user_cpuset("0");
        options.placement().cpu_strategy(PlacementStrategy::PerMachine);
        options.placement().resources_strategy(PlacementResources::Shared);
    });

    auto partitions = make_partitions(options);

    EXPECT_EQ(partitions->host_partitions().size(), 1);
    EXPECT_EQ(partitions->device_partitions().size(), 4);
}

// EngineFactorScenarios
// 1: default engines
// 2: default engines, dedicated main
// 3: default engines, dedicated network
// 4: default engines, multi-node, main/network queues are the same fiber queue
// 5: default engines, multi-node, dedicated main, dedicated network
// 6: 6 core - default engines + services, dedicated_threads, dedicated_fibers (from test_options.cpp)
// 7: 8 core - default engines + services, dedicated_threads, dedicated_fibers (from test_options.cpp)
// 8: 4 core - default engines + services, dedicated_threads, dedicated_fibers (from test_options.cpp)
// 9: factr

TEST_P(TestPartitions, EngineFactoryScenario1)
{
    auto options = make_options([](Options& options) {
        options.topology().user_cpuset("0");
        options.topology().restrict_gpus(true);
        options.placement().cpu_strategy(PlacementStrategy::PerMachine);
        options.placement().resources_strategy(PlacementResources::Shared);
    });

    auto partitions = make_partitions(options);

    EXPECT_EQ(partitions->host_partitions().size(), 1);
    EXPECT_EQ(partitions->device_partitions().size(), 1);

    const auto cpu_sets = partitions->host_partitions().at(0).engine_factory_cpu_sets();

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 2);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 0);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("default").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("main").weight(), 1);

    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 0);
}

TEST_P(TestPartitions, EngineFactoryScenario2)
{
    auto focus = [](Options& options) {
        // options that are the foucs of this test
        options.engine_factories().set_dedicated_main_thread(true);
    };

    auto common = [&focus](Options& options) {
        options.topology().restrict_gpus(true);
        options.placement().cpu_strategy(PlacementStrategy::PerMachine);
        options.placement().resources_strategy(PlacementResources::Shared);
        focus(options);
    };

    auto options_fail = make_options([&common](Options& options) {
        common(options);
        options.topology().user_cpuset("0");
    });

    // not enough cpu cores for a dedicated main thread and at least 1 thread in the default pool
    EXPECT_ANY_THROW(make_partitions(options_fail));

    // expand user_cpuset to the minimum number of required cores
    auto options = make_options([&common](Options& options) {
        common(options);
        options.topology().user_cpuset("0,1");
    });

    auto partitions = make_partitions(options);

    EXPECT_EQ(partitions->host_partitions().size(), 1);
    EXPECT_EQ(partitions->device_partitions().size(), 1);

    const auto cpu_sets = partitions->host_partitions().at(0).engine_factory_cpu_sets();

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 2);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 0);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("default").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("main").weight(), 1);

    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 0);
}

TEST_P(TestPartitions, EngineFactoryScenario3)
{
    auto focus = [](Options& options) {
        // options that are the foucs of this test
        options.architect_url("localhost:13337");
        options.engine_factories().set_dedicated_network_thread(true);
    };

    auto common = [&focus](Options& options) {
        options.topology().restrict_gpus(true);
        options.placement().cpu_strategy(PlacementStrategy::PerMachine);
        options.placement().resources_strategy(PlacementResources::Shared);
        focus(options);
    };

    auto options_fail = make_options([&common](Options& options) {
        common(options);
        options.topology().user_cpuset("0");
    });

    // not enough cpu cores for a dedicated main thread and at least 1 thread in the default pool
    EXPECT_ANY_THROW(make_partitions(options_fail));

    // drop the url, drop the number of cores
    // dedicated network thread requested, but no architect url => no srf_network engine factory
    auto options_no_url = make_options([common](Options& options) {
        common(options);
        options.topology().user_cpuset("0");
        options.architect_url("");
    });

    auto partitions_no_url = make_partitions(options_no_url);
    EXPECT_EQ(partitions_no_url->host_partitions().at(0).engine_factory_cpu_sets().fiber_cpu_sets.size(), 2);

    // expand user_cpuset to the minimum number of required cores
    auto options = make_options([&common](Options& options) {
        common(options);
        options.topology().user_cpuset("0,1");
    });

    auto partitions     = make_partitions(options);
    const auto cpu_sets = partitions->host_partitions().at(0).engine_factory_cpu_sets();

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 3);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 0);
    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 0);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("default").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("main").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("srf_network").weight(), 1);
}

TEST_P(TestPartitions, EngineFactoryScenario4)
{
    auto focus = [](Options& options) {
        // options that are the foucs of this test
        options.architect_url("localhost:13337");
    };

    auto common = [&focus](Options& options) {
        options.topology().restrict_gpus(true);
        options.placement().cpu_strategy(PlacementStrategy::PerMachine);
        options.placement().resources_strategy(PlacementResources::Shared);
        focus(options);
    };

    auto options = make_options([&common](Options& options) {
        common(options);
        options.topology().user_cpuset("0");
    });

    auto partitions     = make_partitions(options);
    const auto cpu_sets = partitions->host_partitions().at(0).engine_factory_cpu_sets();

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 3);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 0);
    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 0);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("default").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("main").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("srf_network").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("main").first(), cpu_sets.fiber_cpu_sets.at("srf_network").first());
}

TEST_P(TestPartitions, EngineFactoryScenario5)
{
    auto focus = [](Options& options) {
        // options that are the foucs of this test
        options.architect_url("localhost:13337");
        options.engine_factories().set_dedicated_main_thread(true);
        options.engine_factories().set_dedicated_network_thread(true);
    };

    auto common = [&focus](Options& options) {
        options.topology().restrict_gpus(true);
        options.placement().cpu_strategy(PlacementStrategy::PerMachine);
        options.placement().resources_strategy(PlacementResources::Shared);
        focus(options);
    };

    auto options = make_options([&common](Options& options) {
        common(options);
        options.topology().user_cpuset("0-2");
    });

    auto partitions     = make_partitions(options);
    const auto cpu_sets = partitions->host_partitions().at(0).engine_factory_cpu_sets();

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 3);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 0);
    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 0);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("default").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("main").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("srf_network").weight(), 1);
    EXPECT_NE(cpu_sets.fiber_cpu_sets.at("main").first(), cpu_sets.fiber_cpu_sets.at("srf_network").first());
}

TEST_P(TestPartitions, EngineFactoryScenario6)
{
    auto focus = [](Options& options) {
        // options that are the foucs of this test
        options.topology().user_cpuset("0-5");
        add_engine_factory_services(options);
        add_engine_factory_dedicated_threads(options);
        add_engine_factory_dedicated_fibers(options);
    };

    auto options = make_options([&focus](Options& options) {
        options.topology().restrict_gpus(true);
        options.placement().cpu_strategy(PlacementStrategy::PerMachine);
        options.placement().resources_strategy(PlacementResources::Shared);
        focus(options);
    });

    auto partitions = make_partitions(options);

    EXPECT_EQ(partitions->host_partitions().size(), 1);
    EXPECT_EQ(partitions->device_partitions().size(), 1);

    const auto cpu_sets = partitions->host_partitions().at(0).engine_factory_cpu_sets();

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 4);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 1);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("dedicated_fibers").weight(), 2);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("services").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("default").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("main").weight(), 1);

    EXPECT_EQ(cpu_sets.thread_cpu_sets.at("dedicated_threads").weight(), 2);

    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 0);
}

TEST_P(TestPartitions, EngineFactoryScenario7)
{
    auto focus = [](Options& options) {
        // options that are the foucs of this test
        options.topology().user_cpuset("0-7");
        add_engine_factory_services(options);
        add_engine_factory_dedicated_threads(options);
        add_engine_factory_dedicated_fibers(options);
    };

    auto options = make_options([&focus](Options& options) {
        options.topology().restrict_gpus(true);
        options.placement().cpu_strategy(PlacementStrategy::PerMachine);
        options.placement().resources_strategy(PlacementResources::Shared);
        focus(options);
    });

    auto partitions     = make_partitions(options);
    const auto cpu_sets = partitions->host_partitions().at(0).engine_factory_cpu_sets();

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 4);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 1);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("dedicated_fibers").weight(), 2);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("services").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("default").weight(), 3);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("main").weight(), 1);

    EXPECT_EQ(cpu_sets.thread_cpu_sets.at("dedicated_threads").weight(), 2);

    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 0);
}

TEST_P(TestPartitions, EngineFactoryScenario8)
{
    auto focus = [](Options& options) {
        // options that are the foucs of this test
        options.topology().user_cpuset("0-3");
        add_engine_factory_services(options);
        add_engine_factory_dedicated_threads(options);
        add_engine_factory_dedicated_fibers(options);
    };

    auto options = make_options([&focus](Options& options) {
        options.topology().restrict_gpus(true);
        options.placement().cpu_strategy(PlacementStrategy::PerMachine);
        options.placement().resources_strategy(PlacementResources::Shared);
        focus(options);
    });

    EXPECT_ANY_THROW(make_partitions(options));
}

TEST_P(TestPartitions, EngineFactoryScenario9)
{
    auto focus = [](Options& options) {
        // options that are the foucs of this test
        options.topology().user_cpuset("0-7");
        add_engine_factory_services(options);
        add_engine_factory_dedicated_threads(options);
        add_engine_factory_dedicated_fibers(options);
        add_engine_factory_shared_fibers(options);
        add_engine_factory_shared_threads_x1(options);
        add_engine_factory_shared_threads_x2(options);
    };

    auto options = make_options([&focus](Options& options) {
        options.topology().restrict_gpus(true);
        options.placement().cpu_strategy(PlacementStrategy::PerMachine);
        options.placement().resources_strategy(PlacementResources::Shared);
        focus(options);
    });

    auto partitions     = make_partitions(options);
    const auto cpu_sets = partitions->host_partitions().at(0).engine_factory_cpu_sets();

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 5);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 3);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("dedicated_fibers").weight(), 2);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("services").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("default").weight(), 1);

    EXPECT_EQ(cpu_sets.thread_cpu_sets.at("dedicated_threads").weight(), 2);

    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 2);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("shared_fibers").weight(), 2);

    EXPECT_EQ(cpu_sets.thread_cpu_sets.at("shared_threads_x1").weight(), 1);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.at("shared_threads_x2").weight(), 2);

    EXPECT_TRUE(cpu_sets.shared_cpus_has_fibers);
}

TEST_P(TestPartitions, EngineFactoryScenario10)
{
    auto focus = [](Options& options) {
        // options that are the foucs of this test
        options.topology().user_cpuset("0-7");
        add_engine_factory_services(options);
        add_engine_factory_dedicated_threads(options);
        add_engine_factory_dedicated_fibers(options);
        add_engine_factory_shared_threads_x1(options);
        add_engine_factory_shared_threads_x2(options);
    };

    auto options = make_options([&focus](Options& options) {
        options.topology().restrict_gpus(true);
        options.placement().cpu_strategy(PlacementStrategy::PerMachine);
        options.placement().resources_strategy(PlacementResources::Shared);
        focus(options);
    });

    auto partitions     = make_partitions(options);
    const auto cpu_sets = partitions->host_partitions().at(0).engine_factory_cpu_sets();

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 4);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 3);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("dedicated_fibers").weight(), 2);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("services").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("default").weight(), 1);

    EXPECT_EQ(cpu_sets.thread_cpu_sets.at("dedicated_threads").weight(), 2);

    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 2);

    EXPECT_FALSE(cpu_sets.shared_cpus_has_fibers);
}

TEST_P(TestPartitions, EngineFactoryScenario11)
{
    auto focus = [](Options& options) {
        // options that are the foucs of this test
        options.topology().user_cpuset("0-7");
        options.engine_factories().set_default_engine_type(runnable::EngineType::Thread);
        add_engine_factory_services(options);
        add_engine_factory_dedicated_threads(options);
        add_engine_factory_dedicated_fibers(options);
        add_engine_factory_shared_threads_x1(options);
        add_engine_factory_shared_threads_x2(options);
    };

    auto options = make_options([&focus](Options& options) {
        options.topology().restrict_gpus(true);
        options.placement().cpu_strategy(PlacementStrategy::PerMachine);
        options.placement().resources_strategy(PlacementResources::Shared);
        focus(options);
    });

    auto partitions     = make_partitions(options);
    const auto cpu_sets = partitions->host_partitions().at(0).engine_factory_cpu_sets();

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 3);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 4);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("services").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("dedicated_fibers").weight(), 2);

    EXPECT_EQ(cpu_sets.thread_cpu_sets.at("default").weight(), 1);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.at("dedicated_threads").weight(), 2);

    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 2);

    EXPECT_FALSE(cpu_sets.shared_cpus_has_fibers);
}

INSTANTIATE_TEST_SUITE_P(Topos, TestPartitions, testing::Values("dgx_a100_station_topology"));

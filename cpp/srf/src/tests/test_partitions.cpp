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

#include <srf/protos/architect.pb.h>
#include <srf/options/topology.hpp>

#include "internal/system/gpu_info.hpp"
#include "internal/system/host_partition.hpp"
#include "internal/system/partitions.hpp"
#include "internal/system/topology.hpp"
#include "srf/core/bitmap.hpp"
#include "srf/options/options.hpp"
#include "srf/options/placement.hpp"

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
        LOG(INFO) << "root_data_path: " << path.str();
        // todo(backlog) - assert the the fixture fie exists
        auto topo_proto = Topology::deserialize_from_file(path.str());
        auto topology   = Topology::Create(options->topology(), topo_proto);
        return std::make_unique<Partitions>(*topology, *options);
    }
};

TEST_P(TestPartitions, Scenario1)
{
    auto options    = make_options();
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
    auto options    = make_options([](Options& options) { options.topology().ignore_dgx_display(false); });
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

INSTANTIATE_TEST_SUITE_P(Topos, TestPartitions, testing::Values("dgx_a100_station_topology"));

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

#include "internal/system/device_info.hpp"
#include "internal/system/gpu_info.hpp"
#include "internal/system/topology.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/options/topology.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <hwloc.h>
#include <hwloc/bitmap.h>
#include <hwloc/nvml.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

using namespace mrc;

// iwyu is getting confused between std::uint32_t and boost::uint32_t
// IWYU pragma: no_include <boost/cstdint.hpp>

class TestTopology : public ::testing::Test
{};

TEST_F(TestTopology, Bitmap)
{
    Bitmap bitmap;
    EXPECT_TRUE(bitmap.empty());

    bitmap.on(3);
    bitmap.on(9);
    EXPECT_EQ(bitmap.weight(), 2);
    EXPECT_TRUE(bitmap.is_set(3));
    EXPECT_TRUE(bitmap.is_set(9));
    EXPECT_FALSE(bitmap.is_set(4));
    EXPECT_FALSE(bitmap.is_set(8));

    bitmap.off(9);
    bitmap.off(3);
    EXPECT_TRUE(bitmap.empty());
    EXPECT_EQ(bitmap.weight(), 0);

    bitmap.only(15);
    EXPECT_EQ(bitmap.weight(), 1);

    bitmap.zero();
    EXPECT_EQ(bitmap.weight(), 0);
    EXPECT_TRUE(bitmap.empty());

    std::vector<int> cpus = {0, 3, 5};
    for (const auto& cpu : cpus)
    {
        bitmap.on(cpu);
    }
    bitmap.for_each_bit([cpus](std::uint32_t i, std::uint32_t bit_index) { EXPECT_EQ(cpus.at(i), bit_index); });

    auto popped = bitmap.pop(2);

    EXPECT_EQ(popped.weight(), 2);
    EXPECT_EQ(bitmap.weight(), 1);

    bitmap.append(popped);
    EXPECT_TRUE(bitmap.str() == "0,3,5");

    CpuSet sub_yes("0,3");
    CpuSet sub_no("0,7");

    EXPECT_TRUE(bitmap.contains(sub_yes));
    EXPECT_FALSE(bitmap.contains(sub_no));
}

TEST_F(TestTopology, TopologyOptions)
{
    auto options  = std::make_unique<TopologyOptions>();
    auto topology = internal::system::Topology::Create(*options);
    EXPECT_GT(topology->cpu_count(), 1);

    auto full_numa = topology->numa_count();

    options->use_process_cpuset(false);
    options->user_cpuset("0");
    topology = internal::system::Topology::Create(*options);
    EXPECT_EQ(topology->cpu_count(), 1);
    EXPECT_EQ(topology->numa_count(), 1);

    EXPECT_TRUE(topology->contains(CpuSet("0")));
    EXPECT_FALSE(topology->contains(CpuSet("1")));

    // impossible cpu_set
    options->user_cpuset("9999999");
    EXPECT_ANY_THROW(topology = internal::system::Topology::Create(*options));

    // should receive a warning
    options->user_cpuset("0,9999999");
    LOG(INFO) << "*** expect a warning below this mark ***";
    topology = internal::system::Topology::Create(*options);
    LOG(INFO) << "*** expect a warning above this mark ***";

    if (full_numa > 1)
    {
        options->restrict_numa_domains(false);
        options->use_process_cpuset(false);
        options->user_cpuset("0");
        topology = internal::system::Topology::Create(*options);
        EXPECT_EQ(topology->cpu_count(), 1);
        EXPECT_EQ(topology->numa_count(), full_numa);
    }
    else
    {
        GTEST_SKIP() << "unable to test numa bindings on non-numa systems";
    }
}

TEST_F(TestTopology, HwlocDev)
{
    hwloc_topology_t topology;
    hwloc_topology_init(&topology);

    auto* device  = internal::system::DeviceInfo::GetHandleById(0);
    auto* cpu_set = hwloc_bitmap_alloc();
    auto rc       = hwloc_nvml_get_device_cpuset(topology, device, cpu_set);
    EXPECT_EQ(rc, 0);

    char* cpuset_string = nullptr;
    hwloc_bitmap_asprintf(&cpuset_string, cpu_set);
    printf("got cpu_set %s for device %d\n", cpuset_string, 0);
    free(cpuset_string);

    auto last = hwloc_bitmap_last(cpu_set);
    LOG(INFO) << "last: " << last;

    auto weight = hwloc_bitmap_weight(cpu_set);
    LOG(INFO) << "weigth: " << weight;

    std::set<int> cpus;
    for (int next = -1;;)
    {
        next = hwloc_bitmap_next(cpu_set, next);
        if (next == -1)
        {
            break;
        }
        cpus.insert(next);
    }

    hwloc_bitmap_free(cpu_set);
    hwloc_topology_destroy(topology);
}

TEST_F(TestTopology, ExportXML)
{
    auto topology = internal::system::Topology::Create();
    auto xml      = topology->export_xml();

    auto pos = xml.find("object type=\"Machine\"");
    EXPECT_NE(pos, std::string::npos);
}

TEST_F(TestTopology, Codable)
{
    auto topology = internal::system::Topology::Create();
    auto encoded  = topology->serialize();
    auto decoded  = internal::system::Topology::Create(encoded);

    EXPECT_EQ(topology->cpu_set().str(), decoded->cpu_set().str());
    EXPECT_EQ(topology->cpu_count(), decoded->cpu_count());
    EXPECT_EQ(topology->core_count(), decoded->core_count());
    EXPECT_EQ(topology->numa_count(), decoded->numa_count());
    EXPECT_EQ(topology->gpu_count(), decoded->gpu_count());

    for (int i = 0; i < topology->numa_count(); ++i)
    {
        EXPECT_EQ(topology->numa_cpuset(i).str(), decoded->numa_cpuset(i).str());
    }

    EXPECT_EQ(topology->gpu_info().size(), decoded->gpu_info().size());

    for (const auto& [id, info] : topology->gpu_info())
    {
        EXPECT_EQ(info.cpu_set().str(), decoded->gpu_info().at(id).cpu_set().str());
        EXPECT_EQ(info.cpustr(), decoded->gpu_info().at(id).cpustr());
        EXPECT_EQ(info.name(), decoded->gpu_info().at(id).name());
        EXPECT_EQ(info.uuid(), decoded->gpu_info().at(id).uuid());
        EXPECT_EQ(info.pcie_bus_id(), decoded->gpu_info().at(id).pcie_bus_id());
    }
}

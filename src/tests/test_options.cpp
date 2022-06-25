/**
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/system/engine_factory_cpu_sets.hpp"
#include "internal/system/topology.hpp"

#include "srf/core/bitmap.hpp"
#include "srf/options/engine_groups.hpp"
#include "srf/options/options.hpp"
#include "srf/options/placement.hpp"
#include "srf/options/topology.hpp"
#include "srf/runnable/types.hpp"

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <utility>

using namespace srf;

class TestOptions : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        {
            EngineFactoryOptions group;
            group.engine_type   = runnable::EngineType::Fiber;
            group.allow_overlap = false;
            group.reusable      = true;
            group.cpu_count     = 1;
            m_options.engine_factories().set_engine_factory_options("services", std::move(group));
        }

        {
            EngineFactoryOptions group;
            group.engine_type   = runnable::EngineType::Thread;
            group.allow_overlap = false;
            group.reusable      = false;
            group.cpu_count     = 2;
            m_options.engine_factories().set_engine_factory_options("dedicated_threads", std::move(group));
        }

        {
            EngineFactoryOptions group;
            group.engine_type   = runnable::EngineType::Fiber;
            group.allow_overlap = false;
            group.reusable      = false;
            group.cpu_count     = 2;
            m_options.engine_factories().set_engine_factory_options("dedicated_fibers", std::move(group));
        }
    }

    void TearDown() override {}

    void initialize()
    {
        m_topology  = Topology::Create(m_options.topology());
        m_placement = Placement::Create(m_options.placement(), m_topology);
    }

    const Options& options() const
    {
        return m_options;
    }

    Options m_options;
    std::shared_ptr<Topology> m_topology;
    std::shared_ptr<Placement> m_placement;
};

// TODO(ryan) - load topologies from files rather than from whatever hardware the CI runner is executing on

TEST_F(TestOptions, ResourceManager6Cpus)
{
    m_options.topology().user_cpuset("0-5");
    m_options.placement().cpu_strategy(PlacementStrategy::PerMachine);

    initialize();

    const auto cpu_sets = generate_launch_control_placement_cpu_sets(options(), m_placement->group(0).cpu_set());

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 4);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 1);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("dedicated_fibers").weight(), 2);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("services").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("default").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("main").weight(), 1);

    EXPECT_EQ(cpu_sets.thread_cpu_sets.at("dedicated_threads").weight(), 2);

    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 0);
}

TEST_F(TestOptions, ResourceManager8Cpus)
{
    m_options.topology().user_cpuset("0-7");
    m_options.placement().cpu_strategy(PlacementStrategy::PerMachine);

    initialize();

    const auto cpu_sets = generate_launch_control_placement_cpu_sets(options(), m_placement->group(0).cpu_set());

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 4);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 1);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("dedicated_fibers").weight(), 2);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("services").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("default").weight(), 3);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("main").weight(), 1);

    EXPECT_EQ(cpu_sets.thread_cpu_sets.at("dedicated_threads").weight(), 2);

    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 0);
}

TEST_F(TestOptions, ResourceManagerNotEnoughCpus)
{
    m_options.topology().user_cpuset("0-3");
    m_options.placement().cpu_strategy(PlacementStrategy::PerMachine);

    initialize();

    EXPECT_ANY_THROW(const auto cpu_sets =
                         generate_launch_control_placement_cpu_sets(options(), m_placement->group(0).cpu_set()));
}

TEST_F(TestOptions, ResourceManagerAddOverlappableGroups)
{
    m_options.topology().user_cpuset("0-7");
    m_options.placement().cpu_strategy(PlacementStrategy::PerMachine);

    {
        EngineFactoryOptions group;
        group.engine_type   = runnable::EngineType::Fiber;
        group.allow_overlap = true;
        group.reusable      = true;
        group.cpu_count     = 2;
        m_options.engine_factories().set_engine_factory_options("shared_fibers", std::move(group));
    }

    {
        EngineFactoryOptions group;
        group.engine_type   = runnable::EngineType::Thread;
        group.allow_overlap = true;
        group.reusable      = true;
        group.cpu_count     = 1;
        m_options.engine_factories().set_engine_factory_options("shared_threads_x1", std::move(group));
    }

    {
        EngineFactoryOptions group;
        group.engine_type   = runnable::EngineType::Thread;
        group.allow_overlap = true;
        group.reusable      = true;
        group.cpu_count     = 2;
        m_options.engine_factories().set_engine_factory_options("shared_threads_x2", std::move(group));
    }

    initialize();

    const auto cpu_sets = generate_launch_control_placement_cpu_sets(options(), m_placement->group(0).cpu_set());

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

TEST_F(TestOptions, ResourceManagerWithArchitectDefaultNetworkEngineFactory)
{
    m_options.topology().user_cpuset("0-7");
    m_options.placement().cpu_strategy(PlacementStrategy::PerMachine);
    m_options.architect_url("localhost:13337");

    {
        EngineFactoryOptions group;
        group.engine_type   = runnable::EngineType::Thread;
        group.allow_overlap = true;
        group.reusable      = true;
        group.cpu_count     = 1;
        m_options.engine_factories().set_engine_factory_options("shared_thread_single", std::move(group));
    }

    {
        EngineFactoryOptions group;
        group.engine_type   = runnable::EngineType::Thread;
        group.allow_overlap = true;
        group.reusable      = true;
        group.cpu_count     = 2;
        m_options.engine_factories().set_engine_factory_options("shared_threads", std::move(group));
    }

    initialize();

    const auto cpu_sets = generate_launch_control_placement_cpu_sets(options(), m_placement->group(0).cpu_set());

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 5);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 3);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("dedicated_fibers").weight(), 2);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("services").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("default").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("srf_network").weight(), 1);

    EXPECT_EQ(cpu_sets.thread_cpu_sets.at("dedicated_threads").weight(), 2);

    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 2);

    EXPECT_TRUE(cpu_sets.shared_cpus_has_fibers);
}

TEST_F(TestOptions, ResourceManagerAddNoFibersInOverlap)
{
    m_options.topology().user_cpuset("0-7");
    m_options.placement().cpu_strategy(PlacementStrategy::PerMachine);

    {
        EngineFactoryOptions group;
        group.engine_type   = runnable::EngineType::Thread;
        group.allow_overlap = true;
        group.reusable      = true;
        group.cpu_count     = 1;
        m_options.engine_factories().set_engine_factory_options("shared_thread_single", std::move(group));
    }

    {
        EngineFactoryOptions group;
        group.engine_type   = runnable::EngineType::Thread;
        group.allow_overlap = true;
        group.reusable      = true;
        group.cpu_count     = 2;
        m_options.engine_factories().set_engine_factory_options("shared_threads", std::move(group));
    }

    initialize();

    const auto cpu_sets = generate_launch_control_placement_cpu_sets(options(), m_placement->group(0).cpu_set());

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 4);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 3);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("dedicated_fibers").weight(), 2);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("services").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("default").weight(), 1);

    EXPECT_EQ(cpu_sets.thread_cpu_sets.at("dedicated_threads").weight(), 2);

    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 2);

    EXPECT_FALSE(cpu_sets.shared_cpus_has_fibers);
}

TEST_F(TestOptions, ResourceManagerDefaultThreadEngineFactory)
{
    m_options.topology().user_cpuset("0-7");
    m_options.placement().cpu_strategy(PlacementStrategy::PerMachine);
    m_options.engine_factories().set_default_engine_type(runnable::EngineType::Thread);

    {
        EngineFactoryOptions group;
        group.engine_type   = runnable::EngineType::Thread;
        group.allow_overlap = true;
        group.reusable      = true;
        group.cpu_count     = 1;
        m_options.engine_factories().set_engine_factory_options("shared_thread_single", std::move(group));
    }

    {
        EngineFactoryOptions group;
        group.engine_type   = runnable::EngineType::Thread;
        group.allow_overlap = true;
        group.reusable      = true;
        group.cpu_count     = 2;
        m_options.engine_factories().set_engine_factory_options("shared_threads", std::move(group));
    }

    initialize();

    const auto cpu_sets = generate_launch_control_placement_cpu_sets(options(), m_placement->group(0).cpu_set());

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.size(), 3);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.size(), 4);

    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("services").weight(), 1);
    EXPECT_EQ(cpu_sets.fiber_cpu_sets.at("dedicated_fibers").weight(), 2);

    EXPECT_EQ(cpu_sets.thread_cpu_sets.at("default").weight(), 1);
    EXPECT_EQ(cpu_sets.thread_cpu_sets.at("dedicated_threads").weight(), 2);

    EXPECT_EQ(cpu_sets.shared_cpus_set.weight(), 2);

    EXPECT_FALSE(cpu_sets.shared_cpus_has_fibers);
}

/*
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

#include "tests/common.hpp"

#include "internal/resources/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/runnable/runnable_resources.hpp"
#include "internal/system/system.hpp"
#include "internal/system/system_provider.hpp"

#include "mrc/core/task_queue.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/placement.hpp"
#include "mrc/types.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace mrc;

// iwyu is getting confused between std::uint32_t and boost::uint32_t
// IWYU pragma: no_include <boost/cstdint.hpp>

class TestResources : public ::testing::Test
{};

TEST_F(TestResources, Lifetime)
{
    auto resources = tests::make_threading_resources();
}

TEST_F(TestResources, GetRuntime)
{
    auto resources = std::make_unique<resources::Manager>(
        system::SystemProvider(tests::make_system([](Options& options) {
            // todo(#114) - propose: this is the default and only option
            options.placement().resources_strategy(PlacementResources::Dedicated);
        })));

    EXPECT_ANY_THROW(resources::Manager::get_resources());
    EXPECT_ANY_THROW(resources::Manager::get_partition());

    resources->partition(0)
        .runnable()
        .main()
        .enqueue([] {
            auto& resources = resources::Manager::get_resources();
            auto& partition = resources::Manager::get_partition();
            EXPECT_EQ(partition.partition_id(), 0);
        })
        .get();
}

TEST_F(TestResources, GetRuntimeShared)
{
    auto resources = std::make_unique<resources::Manager>(
        system::SystemProvider(tests::make_system([](Options& options) {
            // todo(#114) - propose: remove this option entirely
            options.placement().resources_strategy(PlacementResources::Shared);
        })));

    EXPECT_ANY_THROW(resources::Manager::get_resources());
    EXPECT_ANY_THROW(resources::Manager::get_partition());

    resources->partition(0)
        .runnable()
        .main()
        .enqueue([] {
            auto& resources = resources::Manager::get_resources();
            EXPECT_ANY_THROW(resources::Manager::get_partition());
        })
        .get();
}

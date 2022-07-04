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

#include "internal/resources/forward.hpp"
#include "internal/resources/manager.hpp"

#include "srf/channel/forward.hpp"
#include "srf/options/options.hpp"

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <utility>

using namespace srf;
using namespace internal;

// iwyu is getting confused between std::uint32_t and boost::uint32_t
// IWYU pragma: no_include <boost/cstdint.hpp>

class TestResources : public ::testing::Test
{
  protected:
    static std::shared_ptr<system::System> make_system(std::function<void(Options&)> updater = nullptr)
    {
        auto options = std::make_shared<Options>();
        if (updater)
        {
            updater(*options);
        }

        return system::make_system(std::move(options));
    }
};

TEST_F(TestResources, GetRuntime)
{
    auto resources = std::make_unique<internal::resources::Manager>(
        internal::system::SystemProvider(make_system([](Options& options) {
            // todo(#114) - propose: this is the default and only option
            options.placement().resources_strategy(PlacementResources::Dedicated);
        })));

    EXPECT_ANY_THROW(internal::resources::Manager::get_resources());
    EXPECT_ANY_THROW(internal::resources::Manager::get_partition());

    resources->partition(0)
        .runnable()
        .main()
        .enqueue([] {
            auto& resources = internal::resources::Manager::get_resources();
            auto& partition = internal::resources::Manager::get_partition();
            EXPECT_EQ(partition.partition_id(), 0);
        })
        .get();
}

TEST_F(TestResources, GetRuntimeShared)
{
    auto resources = std::make_unique<internal::resources::Manager>(
        internal::system::SystemProvider(make_system([](Options& options) {
            // todo(#114) - propose: remove this option entirely
            options.placement().resources_strategy(PlacementResources::Shared);
        })));

    EXPECT_ANY_THROW(internal::resources::Manager::get_resources());
    EXPECT_ANY_THROW(internal::resources::Manager::get_partition());

    resources->partition(0)
        .runnable()
        .main()
        .enqueue([] {
            auto& resources = internal::resources::Manager::get_resources();
            EXPECT_ANY_THROW(internal::resources::Manager::get_partition());
        })
        .get();
}

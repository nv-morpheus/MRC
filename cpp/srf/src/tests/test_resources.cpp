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

#include "internal/pipeline/types.hpp"
#include "internal/resources/resource_partitions.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/system/system.hpp"
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

        return system::System::make_system(std::move(options));
    }
};

TEST_F(TestResources, LifeCycleSystemResources)
{
    auto system_resources = resources::make_system_resources(make_system());
}

TEST_F(TestResources, LifeCycleResourcePartitions)
{
    auto resource_partitions = resources::make_resource_partitions(make_system());
}

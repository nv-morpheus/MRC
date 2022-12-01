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
#include "internal/control_plane/client/instance.hpp"
#include "internal/network/resources.hpp"
#include "internal/remote_descriptor/manager.hpp"
#include "internal/resources/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/runtime/partition.hpp"
#include "internal/runtime/runtime.hpp"
#include "internal/system/system_provider.hpp"

#include "mrc/codable/codable_protocol.hpp"
#include "mrc/codable/fundamental_types.hpp"  // IWYU pragma: keep
#include "mrc/core/bitmap.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/placement.hpp"
#include "mrc/runtime/remote_descriptor.hpp"
#include "mrc/runtime/remote_descriptor_handle.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <boost/fiber/operations.hpp>
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

using namespace mrc;
using namespace mrc::codable;

class TestRD : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        auto resources = std::make_unique<internal::resources::Manager>(
            internal::system::SystemProvider(make_system([](Options& options) {
                // todo(#114) - propose: remove this option entirely
                options.enable_server(true);
                options.architect_url("localhost:13337");
                options.placement().resources_strategy(PlacementResources::Dedicated);
            })));

        m_runtime = std::make_unique<internal::runtime::Runtime>(std::move(resources));
    }

    void TearDown() override
    {
        m_runtime.reset();
    }

    std::unique_ptr<internal::runtime::Runtime> m_runtime;
};

TEST_F(TestRD, LifeCycle)
{
    m_runtime->partition(0)
        .resources()
        .runnable()
        .main()
        .enqueue([this] {
            auto& rd_manager = m_runtime->partition(0).remote_descriptor_manager();

            EXPECT_EQ(rd_manager.size(), 0);

            std::string test("Hi MRC");
            auto rd = rd_manager.register_object(std::move(test));

            EXPECT_EQ(rd_manager.size(), 1);

            // use the internal implementation to transfer ownership and recreate
            auto handle = internal::remote_descriptor::Manager::unwrap_handle(std::move(rd));
            EXPECT_FALSE(rd);

            // recreate from handle
            auto rd2 = rd_manager.make_remote_descriptor(std::move(handle));
            EXPECT_TRUE(rd2);

            rd.release_ownership();
            rd2.release_ownership();
            EXPECT_EQ(rd_manager.size(), 0);
        })
        .get();
}

TEST_F(TestRD, RemoteRelease)
{
    if (m_runtime->resources().partition_count() < 2)
    {
        GTEST_SKIP() << "this test only works with 2 or more partitions";
    }

    auto f1 = m_runtime->partition(0).resources().network()->control_plane().client().connections().update_future();
    m_runtime->partition(0).resources().network()->control_plane().client().request_update();

    m_runtime->partition(0)
        .resources()
        .runnable()
        .main()
        .enqueue([this] {
            auto& rd_manager_0 = m_runtime->partition(0).remote_descriptor_manager();
            auto& rd_manager_1 = m_runtime->partition(1).remote_descriptor_manager();

            EXPECT_EQ(rd_manager_0.size(), 0);
            EXPECT_EQ(rd_manager_1.size(), 0);

            std::string test("Hi MRC");
            auto rd = rd_manager_0.register_object(std::move(test));

            EXPECT_EQ(rd_manager_0.size(), 1);
            EXPECT_EQ(rd_manager_1.size(), 0);

            auto handle = internal::remote_descriptor::Manager::unwrap_handle(std::move(rd));
            EXPECT_FALSE(rd);

            auto rd2 = rd_manager_1.make_remote_descriptor(std::move(handle));

            EXPECT_EQ(rd_manager_0.size(), 1);
            EXPECT_EQ(rd_manager_1.size(), 0);
            EXPECT_TRUE(rd2);

            rd.release_ownership();
            rd2.release_ownership();

            while (rd_manager_0.size() != 0)
            {
                boost::this_fiber::yield();
            }

            EXPECT_EQ(rd_manager_0.size(), 0);
        })
        .get();
}

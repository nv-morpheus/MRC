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
#include "internal/remote_descriptor/manager.hpp"
#include "internal/remote_descriptor/remote_descriptor.hpp"
#include "internal/resources/manager.hpp"

#include "srf/codable/codable_protocol.hpp"
#include "srf/codable/fundamental_types.hpp"

#include <boost/fiber/operations.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace srf;
using namespace srf::codable;

class TestRD : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_resources = std::make_unique<internal::resources::Manager>(
            internal::system::SystemProvider(make_system([](Options& options) {
                // todo(#114) - propose: remove this option entirely
                options.enable_server(true);
                options.architect_url("localhost:13337");
                options.placement().resources_strategy(PlacementResources::Dedicated);
            })));
    }

    void TearDown() override
    {
        m_resources.reset();
    }

    std::unique_ptr<internal::resources::Manager> m_resources;
};

TEST_F(TestRD, LifeCycle)
{
    m_resources->partition(0)
        .runnable()
        .main()
        .enqueue([this] {
            auto& rd_manager = m_resources->partition(0).network()->remote_descriptor_manager();

            EXPECT_EQ(rd_manager.size(), 0);

            std::string test("Hi SRFer");
            auto rd = rd_manager.register_object(std::move(test));

            EXPECT_EQ(rd_manager.size(), 1);

            auto handle = rd.release_ownership();
            EXPECT_FALSE(rd);

            auto rd2 = rd_manager.take_ownership(std::move(handle));
            EXPECT_TRUE(rd2);

            rd.release();
            rd2.release();
            EXPECT_EQ(rd_manager.size(), 0);
        })
        .get();
}

TEST_F(TestRD, RemoteRelease)
{
    if (m_resources->partition_count() < 2)
    {
        GTEST_SKIP() << "this test only works with 2 or more partitions";
    }

    auto f1 = m_resources->partition(0).network()->control_plane().client().connections().update_future();
    m_resources->partition(0).network()->control_plane().client().request_update();

    m_resources->partition(0)
        .runnable()
        .main()
        .enqueue([this] {
            auto& rd_manager_0 = m_resources->partition(0).network()->remote_descriptor_manager();
            auto& rd_manager_1 = m_resources->partition(1).network()->remote_descriptor_manager();

            EXPECT_EQ(rd_manager_0.size(), 0);
            EXPECT_EQ(rd_manager_1.size(), 0);

            std::string test("Hi SRFer");
            auto rd = rd_manager_0.register_object(std::move(test));

            EXPECT_EQ(rd_manager_0.size(), 1);
            EXPECT_EQ(rd_manager_1.size(), 0);

            auto handle = rd.release_ownership();
            EXPECT_FALSE(rd);

            auto rd2 = rd_manager_1.take_ownership(std::move(handle));
            EXPECT_EQ(rd_manager_0.size(), 1);
            EXPECT_EQ(rd_manager_1.size(), 0);
            EXPECT_TRUE(rd2);

            rd.release();
            rd2.release();

            while (rd_manager_0.size() != 0)
            {
                boost::this_fiber::yield();
            }

            EXPECT_EQ(rd_manager_0.size(), 0);
        })
        .get();
}

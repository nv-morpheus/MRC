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

#include "internal/remote_descriptor/manager.hpp"
#include "internal/remote_descriptor/remote_descriptor.hpp"
#include "internal/resources/manager.hpp"

#include "srf/codable/codable_protocol.hpp"
#include "srf/codable/fundamental_types.hpp"

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
                options.architect_url("localhost:13337");
                options.placement().resources_strategy(PlacementResources::Dedicated);
            })));

        m_rd_manager = std::make_shared<internal::remote_descriptor::Manager>();
    }

    void TearDown() override
    {
        m_resources.reset();
    }

    std::unique_ptr<internal::resources::Manager> m_resources;
    std::shared_ptr<internal::remote_descriptor::Manager> m_rd_manager;
};

TEST_F(TestRD, LifeCycle)
{
    m_resources->partition(0)
        .runnable()
        .main()
        .enqueue([this] {
            EXPECT_EQ(m_rd_manager->size(), 0);

            std::string test("Hi SRFer");
            auto rd = m_rd_manager->register_object(std::move(test));
            EXPECT_EQ(m_rd_manager->size(), 1);

            rd.release();
            EXPECT_EQ(m_rd_manager->size(), 0);
        })
        .get();
}

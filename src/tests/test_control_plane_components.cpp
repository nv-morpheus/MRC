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

#include "internal/control_plane/server/client_instance.hpp"
#include "internal/control_plane/server/tagged_issuer.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace mrc;
using namespace mrc::internal::control_plane;

class TestControlPlaneComponents : public ::testing::Test
{};

struct TaggedObject : public server::Tagged
{
    ~TaggedObject() override = default;
    using server::Tagged::next_tag;
};

class TaggedIssuer : public server::TaggedIssuer
{
  public:
    TaggedIssuer(std::function<void(const TagID& tag)> on_drop) : m_on_drop(std::move(on_drop)) {}
    ~TaggedIssuer() override
    {
        this->drop_all();
    }

    using server::TaggedIssuer::register_instance_id;

  private:
    std::function<void(const TagID& tag)> m_on_drop;
    void do_drop_tag(const TagID& tag) final
    {
        EXPECT_TRUE(m_on_drop);
        m_on_drop(tag);
    };

    void do_issue_update() final {}

    const std::string& service_name() const final
    {
        static std::string name = "TestTaggedIssuer";
        return name;
    }
};

TEST_F(TestControlPlaneComponents, Tagged)
{
    TaggedObject tagged1;
    TaggedObject tagged2;

    EXPECT_EQ(tagged1.upper_bound() + 1, tagged2.lower_bound());

    for (int i = 1; i < UINT16_MAX; i++)
    {
        auto tag = tagged1.next_tag();
        EXPECT_TRUE(tagged1.is_valid_tag(tag));
        EXPECT_FALSE(tagged1.is_issued_tag(tag + 1));
    }

    // we can create UINT16_MAX tags per tagged object
    // this should throw with an overflow
    EXPECT_ANY_THROW(tagged1.next_tag());
}

TEST_F(TestControlPlaneComponents, TaggedIssuer)
{
    std::atomic<std::size_t> counter = 0;
    auto service                     = std::make_unique<TaggedIssuer>([&counter](const TagID& tag) { ++counter; });

    std::vector<TagID> tags;
    tags.push_back(service->register_instance_id(1));
    tags.push_back(service->register_instance_id(2));
    tags.push_back(service->register_instance_id(2));
    tags.push_back(service->register_instance_id(3));
    tags.push_back(service->register_instance_id(3));
    tags.push_back(service->register_instance_id(3));

    EXPECT_EQ(service->tag_count(), 6);
    EXPECT_EQ(service->tag_count_for_instance_id(1), 1);
    EXPECT_EQ(service->tag_count_for_instance_id(2), 2);
    EXPECT_EQ(service->tag_count_for_instance_id(3), 3);
    EXPECT_EQ(counter, 0);

    service->drop_instance(3);
    EXPECT_EQ(service->tag_count(), 3);
    EXPECT_EQ(service->tag_count_for_instance_id(3), 0);
    EXPECT_EQ(counter, 3);

    // remove first recorded tag associated with instance 2
    service->drop_tag(tags[1]);
    EXPECT_EQ(service->tag_count_for_instance_id(1), 1);
    EXPECT_EQ(service->tag_count_for_instance_id(2), 1);
    EXPECT_EQ(counter, 4);

    service->drop_all();
    EXPECT_EQ(service->tag_count(), 0);
    EXPECT_EQ(service->tag_count_for_instance_id(1), 0);
    EXPECT_EQ(service->tag_count_for_instance_id(2), 0);
    EXPECT_EQ(service->tag_count_for_instance_id(3), 0);
    EXPECT_EQ(counter, 6);
}

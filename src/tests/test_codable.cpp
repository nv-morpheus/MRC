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

#include "internal/data_plane/resources.hpp"
#include "internal/resources/manager.hpp"

#include "srf/codable/codable_protocol.hpp"
#include "srf/codable/decode.hpp"
#include "srf/codable/encode.hpp"
#include "srf/codable/encoded_object.hpp"
#include "srf/codable/encoding_options.hpp"
#include "srf/codable/fundamental_types.hpp"
#include "srf/codable/protobuf_message.hpp"
#include "srf/codable/type_traits.hpp"
#include "srf/protos/codable.pb.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

using namespace srf;
using namespace srf::codable;

class CodableObject
{
  public:
    CodableObject()  = default;
    ~CodableObject() = default;

    static CodableObject deserialize(const EncodedObject& buffer, std::size_t /*unused*/)
    {
        return CodableObject();
    }

    void serialize(Encoded<CodableObject>& /*unused*/) {}
};

class CodableObjectWithOptions
{
  public:
    CodableObjectWithOptions()  = default;
    ~CodableObjectWithOptions() = default;

    static CodableObjectWithOptions deserialize(const EncodedObject& encoding, std::size_t /*unused*/)
    {
        return CodableObjectWithOptions();
    }

    void serialize(Encoded<CodableObjectWithOptions>& /*unused*/, const EncodingOptions& opts) {}
};

class CodableViaExternalStruct
{};

namespace srf::codable {

template <>
struct codable_protocol<CodableViaExternalStruct>
{
    void serialize(const CodableViaExternalStruct& /*unused*/, Encoded<CodableViaExternalStruct>& /*unused*/) {}
};

};  // namespace srf::codable

namespace srf::codable {}

struct NotCodableObject
{};

class TestCodable : public ::testing::Test
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
    }

    void TearDown() override
    {
        m_resources.reset();
    }

    std::unique_ptr<internal::resources::Manager> m_resources;
};

TEST_F(TestCodable, Objects)
{
    static_assert(codable::is_encodable<CodableObject>::value, "should be encodable");
    static_assert(codable::is_encodable<CodableObjectWithOptions>::value, "should be encodable");
    static_assert(codable::is_encodable<CodableViaExternalStruct>::value, "should be encodable");
    static_assert(!codable::is_encodable<NotCodableObject>::value, "should NOT be encodable");
    // the following will fail to compile
    // static_assert(codable::is_encodable<NotCodableObject>::value, "should NOT be encodable");

    static_assert(codable::is_decodable<CodableObject>::value, "should be decodable");
    static_assert(codable::is_decodable<CodableObjectWithOptions>::value, "should be decodable");
    static_assert(!codable::is_decodable<CodableViaExternalStruct>::value, "should NOT be decodable");
    static_assert(!codable::is_decodable<NotCodableObject>::value, "should NOT be decodable");
    // the following will fail to compile
    // static_assert(codable::is_decodable<CodableViaExternalStruct>::value, "should NOT be decodable");

    static_assert(is_codable<CodableObject>::value, "fully codable");
    static_assert(is_codable<CodableObjectWithOptions>::value, "fully codable");
    static_assert(!is_codable<CodableViaExternalStruct>::value, "half codable");
    static_assert(!is_codable<NotCodableObject>::value, "not codable");
}

TEST_F(TestCodable, String)
{
    static_assert(is_codable<std::string>::value, "should be codable");

    m_resources->partition(0)
        .runnable()
        .main()
        .enqueue([this] {
            std::string str = "Hello SRF";
            auto str_block  = m_resources->partition(0).network()->data_plane().registration_cache().lookup(str.data());
            EXPECT_FALSE(str_block);

            auto encoding = encode(str);
            auto decoding = decode<std::string>(*encoding);
            EXPECT_STREQ(str.c_str(), decoding.c_str());

            // test to ensure that the unregistered string got copied to an internal registered buffer
            auto view = encoding->memory_block(0);
            auto view_block =
                m_resources->partition(0).network()->data_plane().registration_cache().lookup(view.data());
            EXPECT_TRUE(view_block);
        })
        .get();
}

TEST_F(TestCodable, Double)
{
    static_assert(is_codable<double>::value, "should be codable");

    m_resources->partition(0)
        .runnable()
        .main()
        .enqueue([] {
            double pi     = 3.14159;
            auto encoding = encode(pi);
            auto decoding = decode<double>(*encoding);

            EXPECT_DOUBLE_EQ(pi, decoding);
        })
        .get();
}

TEST_F(TestCodable, Composite)
{
    static_assert(is_codable<std::string>::value, "should be codable");
    static_assert(is_codable<std::uint64_t>::value, "should be codable");

    m_resources->partition(0)
        .runnable()
        .main()
        .enqueue([] {
            std::string str   = "Hello Srf";
            std::uint64_t ans = 42;

            EncodedObject encoding;

            encode(str, encoding);
            encode(ans, encoding);

            EXPECT_EQ(encoding.object_count(), 2);
            EXPECT_EQ(encoding.descriptor_count(), 2);

            auto decoded_str = decode<std::string>(encoding, 0);
            auto decoded_ans = decode<std::uint64_t>(encoding, 1);

            EXPECT_STREQ(str.c_str(), decoded_str.c_str());
            EXPECT_EQ(ans, decoded_ans);
        })
        .get();
}

TEST_F(TestCodable, EncodedObjectProto)
{
    static_assert(codable::is_encodable<srf::codable::protos::EncodedObject>::value, "should be encodable");
    static_assert(codable::is_decodable<srf::codable::protos::EncodedObject>::value, "should be decodable");
    static_assert(is_codable<srf::codable::protos::EncodedObject>::value, "should be codable");
}

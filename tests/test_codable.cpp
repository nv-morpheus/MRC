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

#include "test_mrc.hpp"  // IWYU pragma: associated

#include "mrc/codable/codable_protocol.hpp"
#include "mrc/codable/decode.hpp"
#include "mrc/codable/encode.hpp"
#include "mrc/codable/encoded_object.hpp"
#include "mrc/codable/encoding_options.hpp"
#include "mrc/codable/fundamental_types.hpp"
#include "mrc/codable/protobuf_message.hpp"
#include "mrc/codable/type_traits.hpp"
#include "mrc/protos/codable.pb.h"

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

using namespace codable;

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

namespace mrc::codable {

template <>
struct codable_protocol<CodableViaExternalStruct>
{
    void serialize(const CodableViaExternalStruct& /*unused*/, Encoded<CodableViaExternalStruct>& /*unused*/) {}
};

};  // namespace mrc::codable

namespace mrc::codable {}

struct NotCodableObject
{};

TEST_CLASS(Codable);

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

    std::string str = "Hello MRC";
    auto encoding   = encode(str);
    auto decoding   = decode<std::string>(*encoding);

    EXPECT_STREQ(str.c_str(), decoding.c_str());
}

TEST_F(TestCodable, Double)
{
    static_assert(is_codable<double>::value, "should be codable");

    double pi     = 3.14159;
    auto encoding = encode(pi);
    auto decoding = decode<double>(*encoding);

    EXPECT_DOUBLE_EQ(pi, decoding);
}

TEST_F(TestCodable, Composite)
{
    static_assert(is_codable<std::string>::value, "should be codable");
    static_assert(is_codable<std::uint64_t>::value, "should be codable");

    std::string str   = "Hello MRC";
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
}

TEST_F(TestCodable, EncodedObjectProto)
{
    static_assert(codable::is_encodable<protos::EncodedObject>::value, "should be encodable");
    static_assert(codable::is_decodable<protos::EncodedObject>::value, "should be decodable");
    static_assert(is_codable<protos::EncodedObject>::value, "should be codable");
}

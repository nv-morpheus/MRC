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
#include "internal/network/resources.hpp"
#include "internal/resources/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/runtime/partition.hpp"
#include "internal/runtime/runtime.hpp"
#include "internal/system/system_provider.hpp"
#include "internal/ucx/registration_cache.hpp"

#include "mrc/codable/api.hpp"
#include "mrc/codable/codable_protocol.hpp"
#include "mrc/codable/decode.hpp"
#include "mrc/codable/encode.hpp"
#include "mrc/codable/encoding_options.hpp"
#include "mrc/codable/fundamental_types.hpp"  // IWYU pragma: keep
#include "mrc/codable/protobuf_message.hpp"   // IWYU pragma: keep
#include "mrc/codable/type_traits.hpp"
#include "mrc/core/bitmap.hpp"
#include "mrc/memory/buffer.hpp"
#include "mrc/memory/codable/buffer.hpp"  // IWYU pragma: keep
#include "mrc/options/options.hpp"
#include "mrc/options/placement.hpp"
#include "mrc/protos/codable.pb.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

using namespace mrc;
using namespace mrc::codable;

class CodableObject
{
  public:
    CodableObject()  = default;
    ~CodableObject() = default;

    static CodableObject deserialize(const Decoder<CodableObject>& buffer, std::size_t /*unused*/)
    {
        return {};
    }

    void serialize(Encoder<CodableObject>& /*unused*/) {}
};

class CodableObjectWithOptions
{
  public:
    CodableObjectWithOptions()  = default;
    ~CodableObjectWithOptions() = default;

    static CodableObjectWithOptions deserialize(const Decoder<CodableObjectWithOptions>& encoding,
                                                std::size_t /*unused*/)
    {
        return {};
    }

    void serialize(Encoder<CodableObjectWithOptions>& /*unused*/, const EncodingOptions& opts) {}
};

class CodableViaExternalStruct
{};

namespace mrc::codable {

template <>
struct codable_protocol<CodableViaExternalStruct>
{
    void serialize(const CodableViaExternalStruct& /*unused*/, Encoder<CodableViaExternalStruct>& /*unused*/) {}
};

};  // namespace mrc::codable

namespace mrc::codable {}

struct NotCodableObject
{};

class TestCodable : public ::testing::Test
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
    auto str_block =
        m_runtime->partition(0).resources().network()->data_plane().registration_cache().lookup(str.data());
    EXPECT_FALSE(str_block);

    auto encodable_storage = m_runtime->partition(0).make_codable_storage();

    encode(str, *encodable_storage);
    EXPECT_EQ(encodable_storage->descriptor_count(), 1);

    auto decoded_str = decode<std::string>(*encodable_storage);
    EXPECT_STREQ(str.c_str(), decoded_str.c_str());
}

TEST_F(TestCodable, Buffer)
{
    static_assert(is_codable<mrc::memory::buffer>::value, "should be codable");

    // std::string str = "Hello MRC";
    // auto str_block  = m_runtime->partition(0).network()->data_plane().registration_cache().lookup(str.data());
    // EXPECT_FALSE(str_block);

    // internal::remote_descriptor::EncodedObject encoded_object(m_runtime->partition(0));

    // encode(str, encoded_object);
    // EXPECT_EQ(encoded_object.descriptor_count(), 1);

    // auto decoded_str = decode<std::string>(encoded_object);
    // EXPECT_STREQ(str.c_str(), decoded_str.c_str());
}

// TEST_F(TestCodable, Double)
// {
//     static_assert(is_codable<double>::value, "should be codable");

//     m_runtime->partition(0)
//         .runnable()
//         .main()
//         .enqueue([] {
//             double pi     = 3.14159;
//             auto encoding = encode(pi);
//             auto decoding = decode<double>(*encoding);

//             EXPECT_DOUBLE_EQ(pi, decoding);
//         })
//         .get();
// }

// TEST_F(TestCodable, Composite)
// {
//     static_assert(is_codable<std::string>::value, "should be codable");
//     static_assert(is_codable<std::uint64_t>::value, "should be codable");

//     m_runtime->partition(0)
//         .runnable()
//         .main()
//         .enqueue([] {
//             std::string str   = "Hello Mrc";
//             std::uint64_t ans = 42;

//             EncodedObject encoding;

//             encode(str, encoding);
//             encode(ans, encoding);

//             EXPECT_EQ(encoding.object_count(), 2);
//             EXPECT_EQ(encoding.descriptor_count(), 2);

//             auto decoded_str = decode<std::string>(encoding, 0);
//             auto decoded_ans = decode<std::uint64_t>(encoding, 1);

//             EXPECT_STREQ(str.c_str(), decoded_str.c_str());
//             EXPECT_EQ(ans, decoded_ans);
//         })
//         .get();
// }

TEST_F(TestCodable, EncodedObjectProto)
{
    static_assert(codable::is_encodable<mrc::codable::protos::EncodedObject>::value, "should be encodable");
    static_assert(codable::is_decodable<mrc::codable::protos::EncodedObject>::value, "should be decodable");
    static_assert(is_codable<mrc::codable::protos::EncodedObject>::value, "should be codable");
}

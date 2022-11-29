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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
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

int random_number()
{
    return (std::rand() % 50 + 1);
}

void populate(int size, int* ptr)
{
    std::srand(unsigned(std::time(nullptr)));
    std::generate(ptr, ptr + size - 1, random_number);
}

TEST_F(TestCodable, Buffer)
{
    static_assert(is_codable<mrc::memory::buffer>::value, "should be codable");

    // Uncomment when local copy is working!
    // auto encodable_storage = m_runtime->partition(0).make_codable_storage();

    // size_t int_count = 100;

    // auto buffer = m_runtime->partition(0).resources().host().make_buffer(int_count * sizeof(int));

    // populate(int_count, static_cast<int*>(buffer.data()));

    // encode(buffer, *encodable_storage);
    // EXPECT_EQ(encodable_storage->descriptor_count(), 1);

    // auto decoding = decode<mrc::memory::buffer>(*encodable_storage);

    // int* input_start  = static_cast<int*>(buffer.data());
    // int* output_start = static_cast<int*>(decoding.data());

    // EXPECT_TRUE(std::equal(input_start, input_start + int_count, output_start));
}

TEST_F(TestCodable, Double)
{
    static_assert(is_codable<double>::value, "should be codable");

    auto encodable_storage = m_runtime->partition(0).make_codable_storage();

    double pi = 3.14159;

    encode(pi, *encodable_storage);
    EXPECT_EQ(encodable_storage->descriptor_count(), 1);

    auto decoding = decode<double>(*encodable_storage);
    EXPECT_DOUBLE_EQ(pi, decoding);
}

TEST_F(TestCodable, Composite)
{
    static_assert(is_codable<std::string>::value, "should be codable");
    static_assert(is_codable<std::uint64_t>::value, "should be codable");

    std::string str   = "Hello Mrc";
    std::uint64_t ans = 42;

    auto encodable_storage = m_runtime->partition(0).make_codable_storage();

    encode(str, *encodable_storage);
    encode(ans, *encodable_storage);

    EXPECT_EQ(encodable_storage->object_count(), 2);
    EXPECT_EQ(encodable_storage->descriptor_count(), 2);

    auto decoded_str = decode<std::string>(*encodable_storage, 0);
    auto decoded_ans = decode<std::uint64_t>(*encodable_storage, 1);

    EXPECT_STREQ(str.c_str(), decoded_str.c_str());
    EXPECT_EQ(ans, decoded_ans);
}

TEST_F(TestCodable, EncodedObjectProto)
{
    static_assert(codable::is_encodable<mrc::codable::protos::EncodedObject>::value, "should be encodable");
    static_assert(codable::is_decodable<mrc::codable::protos::EncodedObject>::value, "should be decodable");
    static_assert(is_codable<mrc::codable::protos::EncodedObject>::value, "should be codable");
}

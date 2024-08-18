/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/data_plane/data_plane_resources.hpp"
#include "internal/network/network_resources.hpp"
#include "internal/remote_descriptor/storage.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/runtime/partition_runtime.hpp"
#include "internal/runtime/runtime.hpp"
#include "internal/system/system.hpp"
#include "internal/system/system_provider.hpp"
#include "internal/ucx/registration_cache.hpp"

#include "mrc/codable/api.hpp"
#include "mrc/codable/codable_protocol.hpp"
#include "mrc/codable/decode.hpp"
#include "mrc/codable/encode.hpp"
#include "mrc/codable/fundamental_types.hpp"  // IWYU pragma: keep
#include "mrc/codable/protobuf_message.hpp"   // IWYU pragma: keep
#include "mrc/memory/codable/buffer.hpp"  // IWYU pragma: keep
#include "mrc/options/options.hpp"
#include "mrc/options/placement.hpp"
#include "mrc/protos/codable.pb.h"  // IWYU pragma: keep

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

namespace mrc::codable {
class EncodingOptions;
}  // namespace mrc::codable
namespace mrc::memory {
class buffer;
}  // namespace mrc::memory

// IWYU pragma: no_forward_declare mrc::codable::codable_protocol

using namespace mrc;
using namespace mrc::codable;

class CodableObject
{
  public:
    CodableObject()  = default;
    ~CodableObject() = default;

    static CodableObject deserialize(const mrc::codable::Decoder2<CodableObject>& decoder)
    {
        return {};
    }

    void serialize(mrc::codable::Encoder2<CodableObject>& encoder) const {}
};

namespace mrc::codable {}

struct NotCodableObject
{};

class TestCodable : public ::testing::Test {};

TEST_F(TestCodable, Objects)
{
    static_assert(codable::encodable<CodableObject>, "should be encodable");
    static_assert(!codable::encodable<NotCodableObject>, "should NOT be encodable");

    static_assert(codable::decodable<CodableObject>, "should be decodable");
    static_assert(!codable::decodable<NotCodableObject>, "should NOT be decodable");
}

TEST_F(TestCodable, String)
{
    static_assert(codable::encodable<std::string>, "should be encodable");
    static_assert(codable::decodable<std::string>, "should be decodable");

    std::string str = "Hello MRC";

    std::unique_ptr<DescriptorObjectHandler> encoded_obj = encode2(str);

    auto decoded_str = decode2<std::string>(*encoded_obj);
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
    // static_assert(codable::encodable<mrc::memory::buffer>, "should be encodable");
    // static_assert(codable::decodable<mrc::memory::buffer>, "should be decodable");

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
    static_assert(codable::encodable<double>, "should be encodable");
    static_assert(codable::decodable<double>, "should be decodable");

    double pi = 3.14159;

    std::unique_ptr<DescriptorObjectHandler> encoded_obj = encode2(pi);

    auto decoding = decode2<double>(*encoded_obj);
    EXPECT_DOUBLE_EQ(pi, decoding);
}

TEST_F(TestCodable, Composite)
{
    static_assert(codable::encodable<std::string>, "should be encodable");
    static_assert(codable::decodable<std::string>, "should be decodable");

    static_assert(codable::encodable<std::uint64_t>, "should be encodable");
    static_assert(codable::decodable<std::uint64_t>, "should be decodable");

    std::string str   = "Hello Mrc";
    std::uint64_t ans = 42;

    std::unique_ptr<DescriptorObjectHandler> encoded_obj1 = encode2(str);
    std::unique_ptr<DescriptorObjectHandler> encoded_obj2 = encode2(ans);

    auto decoded_str = decode2<std::string>(*encoded_obj1);
    auto decoded_ans = decode2<std::uint64_t>(*encoded_obj2);

    EXPECT_STREQ(str.c_str(), decoded_str.c_str());
    EXPECT_EQ(ans, decoded_ans);
}

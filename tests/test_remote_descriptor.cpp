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

#include "./test_srf.hpp"  // IWYU pragma: associated

#include "srf/codable/codable_protocol.hpp"
#include "srf/codable/encode.hpp"
#include "srf/codable/encoded_object.hpp"
#include "srf/codable/encoding_options.hpp"
#include "srf/codable/type_traits.hpp"
#include "srf/core/resources.hpp"
#include "srf/core/tensor.hpp"
#include "srf/memory/block.hpp"
#include "srf/memory/memory_kind.hpp"
#include "srf/memory/resources/device/cuda_malloc_resource.hpp"
#include "srf/memory/resources/host/pinned_memory_resource.hpp"
#include "srf/protos/remote_descriptor.pb.h"
#include "srf/protos/tensor_meta_data.pb.h"
#include "srf/utils/thread_local_shared_pointer.hpp"
#include "srf/utils/type_utils.hpp"

#include <glog/logging.h>
#include <google/protobuf/any.pb.h>
#include <gtest/gtest.h>

#include <memory>
#include <ostream>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>
// IWYU pragma: no_forward_declare srf::codable

namespace srf {

class TensorDescriptorTestObject
{
  public:
    TensorDescriptorTestObject() :
      m_dtype(DataType::create<float>()),
      m_block({reinterpret_cast<void*>(0xDEADBEEF), 1024 * 1024 * 1024, memory::memory_kind_type::device}),
      m_shape({3, 100, 100})
    {}

  private:
    DataType m_dtype;
    memory::block m_block;
    std::vector<TensorIndex> m_shape;
    std::vector<TensorIndex> m_stride;

    friend codable::codable_protocol<TensorDescriptorTestObject>;
};

class TestThreadLocalResources : public core::Resources
{
  public:
    TestThreadLocalResources() :
      m_host_view(std::make_shared<memory::pinned_memory_resource>()),
      m_device_view(std::make_shared<memory::cuda_malloc_resource>(0))
    {}
    ~TestThreadLocalResources() override = default;

    host_view_t host_resource_view() override
    {
        return m_host_view;
    }
    device_view_t device_resource_view() override
    {
        return m_device_view;
    }

  private:
    host_view_t m_host_view;
    device_view_t m_device_view;
};

template <typename T>
struct codable::codable_protocol<T, std::enable_if_t<std::is_same_v<T, TensorDescriptorTestObject>>>
{
    static void serialize(const T& t, Encoded<T>& encoded, const EncodingOptions& opts)
    {
        auto guard = encoded.acquire_encoding_context();
        // todo(ryan) - determine if memory is registered, if not, use the copy path
        auto data_idx = encoded.add_memory_block(t.m_block);
        ::srf::protos::meta_data::TensorMetaData meta;
        meta.set_dtype(t.m_dtype.type_str());
        auto* shape = meta.mutable_shape();
        for (const auto& idx : t.m_shape)
        {
            shape->Add(idx);
        }
        auto* stride = meta.mutable_stride();
        for (const auto& idx : t.m_stride)
        {
            stride->Add(idx);
        }
        encoded.add_meta_data(meta);
    }

    static T deserialize(const EncodedObject& encoded, std::size_t object_idx)
    {
        DCHECK_EQ(std::type_index(typeid(T)).hash_code(), encoded.type_index_hash_for_object(object_idx));
        auto idx = encoded.start_idx_for_object(object_idx);
        return T();
    }
};

}  // namespace srf

class TestRemoteDescriptor : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        utils::ThreadLocalSharedPointer<core::Resources>::set(std::make_shared<TestThreadLocalResources>());
    }
    void TearDown() override
    {
        utils::ThreadLocalSharedPointer<core::Resources>::set(nullptr);
    }
};

TEST_F(TestRemoteDescriptor, Protos)
{
    protos::RemoteDescriptor rd;
    protos::meta_data::TensorMetaData tmd;

    auto* shape = tmd.mutable_shape();
    shape->Add(3);
    shape->Add(320);
    shape->Add(320);
    tmd.set_dtype("f8");

    rd.mutable_meta_data()->PackFrom(tmd);

    protos::meta_data::TensorMetaData unpacked_tmd;
    auto status = rd.meta_data().UnpackTo(&unpacked_tmd);
    EXPECT_TRUE(status);
    EXPECT_EQ(unpacked_tmd.shape_size(), 3);
}

TEST_F(TestRemoteDescriptor, ObjectWithMetaData)
{
    TensorDescriptorTestObject obj;

    static_assert(codable::is_codable<decltype(obj)>::value, "should be true");

    codable::EncodedObject encoded;
    codable::encode(obj, encoded);

    EXPECT_EQ(encoded.object_count(), 1);
    EXPECT_EQ(encoded.descriptor_count(), 2);

    auto rd = encoded.memory_block(0);

    void* expected_addr = reinterpret_cast<void*>(0xDEADBEEF);
    EXPECT_EQ(rd.data(), expected_addr);
    EXPECT_EQ(rd.bytes(), 1024 * 1024 * 1024);

    auto meta = encoded.meta_data<protos::meta_data::TensorMetaData>(1);
    EXPECT_EQ(meta.shape_size(), 3);
    EXPECT_EQ(meta.stride_size(), 0);
}

class EncodedObjectTester : public codable::EncodedObject
{
    using base_t = codable::EncodedObject;

  public:
    using base_t::EncodedObject;

    using base_t::decode_descriptor;
    using base_t::encode_descriptor;

    using base_t::add_device_buffer;
    using base_t::add_host_buffer;
    using base_t::add_memory_block;
    using base_t::add_meta_data;

    using base_t::ContextGuard;
};

TEST_F(TestRemoteDescriptor, v2_EncodedObjectStaticMethods)
{
    EncodedObjectTester obj;
    EncodedObjectTester::ContextGuard guard(obj, std::type_index(typeid(obj)));

    // todo(ryan) - tensor, matx and xtensor future issue
    // replace the following with a tensor object
    protos::meta_data::TensorMetaData tmd;
    auto* shape = tmd.mutable_shape();
    shape->Add(3);
    shape->Add(320);
    shape->Add(320);
    tmd.set_dtype("f8");
    memory::const_block block(reinterpret_cast<void*>(0xDEADBEEF), 1024, memory::memory_kind_type::device);

    obj.add_memory_block(block);
    obj.add_meta_data(tmd);

    EXPECT_EQ(obj.object_count(), 1);
    EXPECT_EQ(obj.descriptor_count(), 2);
}

TEST_F(TestRemoteDescriptor, v2_EncodedObjectwithHostAndDeviceBuffer)
{
    EncodedObjectTester obj;
    EncodedObjectTester::ContextGuard guard(obj, std::type_index(typeid(obj)));

    auto hidx = obj.add_host_buffer(1024);
    auto didx = obj.add_device_buffer(2048);

    EXPECT_EQ(obj.object_count(), 1);
    EXPECT_EQ(obj.descriptor_count(), 2);

    auto hv = obj.memory_block(hidx);
    auto dv = obj.memory_block(didx);

    EXPECT_EQ(hv.bytes(), 1024);
    EXPECT_EQ(dv.bytes(), 2048);
}

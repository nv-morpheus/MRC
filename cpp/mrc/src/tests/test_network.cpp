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

#include "tests/common.hpp"

#include "internal/codable/codable_storage.hpp"
#include "internal/control_plane/client.hpp"
#include "internal/control_plane/client/connections_manager.hpp"
#include "internal/control_plane/client/instance.hpp"
#include "internal/data_plane/client.hpp"
#include "internal/data_plane/data_plane_resources.hpp"
#include "internal/data_plane/request.hpp"
#include "internal/data_plane/server.hpp"
#include "internal/data_plane/tags.hpp"
#include "internal/memory/device_resources.hpp"
#include "internal/memory/host_resources.hpp"
#include "internal/memory/transient_pool.hpp"
#include "internal/network/network_resources.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/runnable/runnable_resources.hpp"
#include "internal/system/system.hpp"
#include "internal/system/system_provider.hpp"
#include "internal/ucx/memory_block.hpp"
#include "internal/ucx/registration_cache.hpp"

#include "mrc/codable/codable_protocol.hpp"
#include "mrc/codable/decode.hpp"
#include "mrc/codable/encode.hpp"
#include "mrc/codable/fundamental_types.hpp"
#include "mrc/coroutines/sync_wait.hpp"
#include "mrc/edge/edge_builder.hpp"
#include "mrc/memory/adaptors.hpp"
#include "mrc/memory/buffer.hpp"
#include "mrc/memory/literals.hpp"
#include "mrc/memory/memory_block_provider.hpp"
#include "mrc/memory/memory_kind.hpp"
#include "mrc/memory/resources/arena_resource.hpp"
#include "mrc/memory/resources/host/malloc_memory_resource.hpp"
#include "mrc/memory/resources/host/pinned_memory_resource.hpp"
#include "mrc/memory/resources/logging_resource.hpp"
#include "mrc/memory/resources/memory_resource.hpp"
#include "mrc/node/operators/router.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/placement.hpp"
#include "mrc/options/resources.hpp"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runnable/runner.hpp"
#include "mrc/runtime/remote_descriptor.hpp"
#include "mrc/types.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <rxcpp/rx.hpp>
#include <sys/types.h>
#include <ucs/memory/memory_type.h>
#include <ucxx/api.h>

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <thread>
#include <utility>

using namespace mrc;
using namespace mrc::memory::literals;

class DataPlaneResources2Tester : public data_plane::DataPlaneResources2
{
  public:
    std::shared_ptr<runtime::Descriptor2> get_descriptor(uint64_t object_id)
    {
        return m_descriptor_by_id.size() ? m_descriptor_by_id[object_id][0] : nullptr;
    }
};

class TestNetwork : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_resources = std::make_unique<DataPlaneResources2Tester>();

        m_resources->set_instance_id(42);

        m_loopback_endpoint = m_resources->create_endpoint(m_resources->address(), m_resources->get_instance_id());
    }

    void TearDown() override
    {
        m_resources.reset();
    }

    void wait_requests(const std::vector<std::shared_ptr<ucxx::Request>>& requests)
    {
        auto remainingRequests = requests;
        while (!remainingRequests.empty())
        {
            auto updatedRequests = std::exchange(remainingRequests, decltype(remainingRequests)());
            for (auto const& r : updatedRequests)
            {
                m_resources->progress();

                if (!r->isCompleted())
                    remainingRequests.push_back(r);
                else
                    r->checkError();
            }
        }
    }

    std::unique_ptr<DataPlaneResources2Tester> m_resources;
    std::shared_ptr<ucxx::Endpoint> m_loopback_endpoint;
};

/**
 * Serialization and deserialization methods for vector objects allocated on Host or Device memory.
 */
namespace mrc::codable {
template <typename T>
struct codable_protocol<std::vector<T>>
{
    static void serialize(const std::vector<T>& obj,
                          mrc::codable::Encoder2<std::vector<T>>& encoder)
    {
        // First put in the size
        mrc::codable::encode2(obj.size(), encoder);

        if constexpr (std::is_fundamental_v<T>)
        {
            // Since these are fundamental types, just encode in a single memory block
            encoder.write_descriptor({obj.data(), obj.size() * sizeof(T), memory::memory_kind::host});
        }
        else
        {
            // Now encode each object
            for (const auto& o : obj)
            {
                mrc::codable::encode2(o, encoder);
            }
        }
    }

    static std::vector<T> deserialize(const Decoder2<std::vector<T>>& decoder)
    {
        // DCHECK_EQ(std::type_index(typeid(std::vector<T>)).hash_code(),
        // decoder.type_index_hash_for_object(object_idx));

        auto count = mrc::codable::decode2<size_t>(decoder);

        auto object = std::vector<T>(count);

        decoder.read_descriptor({object.data(), count * sizeof(T), memory::memory_kind::host});

        return object;
    }
};

template <>
struct codable_protocol<unsigned char*>
{
    static void serialize(const unsigned char* obj,
                          mrc::codable::Encoder2<unsigned char*>& encoder)
    {
        // Since unsigned char* does not carry indicator of size, specify a fixed size for testing purposes
        size_t size = 64_KiB;
        mrc::codable::encode2(size, encoder);

        encoder.write_descriptor({obj, size * sizeof(unsigned char), memory::memory_kind::device});
    }

    static unsigned char* deserialize(const Decoder2<unsigned char*>& decoder)
    {
        size_t size = mrc::codable::decode2<size_t>(decoder);

        unsigned char* object;
        cudaMalloc((void**)&object, size * sizeof(unsigned char));

        decoder.read_descriptor({object, size * sizeof(unsigned char), memory::memory_kind::device});

        return object;
    }
};

}  // namespace mrc::codable

class TransferObject
{
  public:
    TransferObject() = default;
    TransferObject(std::string name, int value, std::vector<u_int8_t>&& data) :
      m_name(std::move(name)),
      m_value(value),
      m_data(std::move(data))
    {}

    bool operator==(const TransferObject& other) const
    {
        return m_name == other.m_name && m_value == other.m_value && m_data == other.m_data;
    }

    void serialize(mrc::codable::Encoder2<TransferObject>& encoder) const
    {
        mrc::codable::encode2(m_name, encoder);
        mrc::codable::encode2(m_value, encoder);
        mrc::codable::encode2(m_data, encoder);
    }

    static TransferObject deserialize(const mrc::codable::Decoder2<TransferObject>& decoder)
    {
        TransferObject obj;

        obj.m_name  = mrc::codable::decode2<std::string, TransferObject>(decoder);
        obj.m_value = mrc::codable::decode2<int, TransferObject>(decoder);
        obj.m_data  = mrc::codable::decode2<std::vector<u_int8_t>, TransferObject>(decoder);

        return obj;
    }

  private:
    std::string m_name;
    int m_value{0};

    std::vector<u_int8_t> m_data;
};

class ComplexObject
{
  public:
    ComplexObject() = default;
    ComplexObject(std::string name, int value, TransferObject obj, std::vector<u_int8_t>&& data) :
      m_name(std::move(name)),
      m_value(value),
      m_obj(std::move(obj)),
      m_data(std::move(data))
    {}

    bool operator==(const ComplexObject& other) const
    {
        return m_name == other.m_name && m_value == other.m_value && m_obj == other.m_obj && m_data == other.m_data;
    }

    void serialize(mrc::codable::Encoder2<ComplexObject>& encoder) const
    {
        mrc::codable::encode2(m_name, encoder);
        mrc::codable::encode2(m_value, encoder);
        mrc::codable::encode2(m_obj, encoder);
        mrc::codable::encode2(m_data, encoder);
    }

    static ComplexObject deserialize(const mrc::codable::Decoder2<ComplexObject>& decoder)
    {
        ComplexObject obj;

        obj.m_name  = mrc::codable::decode2<std::string, ComplexObject>(decoder);
        obj.m_value = mrc::codable::decode2<int, ComplexObject>(decoder);
        obj.m_obj   = mrc::codable::decode2<TransferObject, ComplexObject>(decoder);
        obj.m_data  = mrc::codable::decode2<std::vector<u_int8_t>, ComplexObject>(decoder);

        return obj;
    }

  private:
    std::string m_name;
    int m_value{0};
    TransferObject m_obj;
    std::vector<u_int8_t> m_data;
};

TEST_F(TestNetwork, Arena)
{
    std::shared_ptr<mrc::memory::memory_resource> mr;
    auto pinned  = std::make_shared<mrc::memory::pinned_memory_resource>();
    mr           = pinned;
    auto logging = mrc::memory::make_shared_resource<mrc::memory::logging_resource>(mr, "pinned", 10);
    auto arena   = mrc::memory::make_shared_resource<mrc::memory::arena_resource>(logging, 128_MiB, 512_MiB);
    auto f       = mrc::memory::make_shared_resource<mrc::memory::logging_resource>(arena, "arena", 10);

    auto* ptr = f->allocate(1024);
    f->deallocate(ptr, 1024);
}

// TEST_F(TestNetwork, ResourceManager)
// {
//     // using options.placement().resources_strategy(PlacementResources::Shared)
//     // will test if cudaSetDevice is being properly called by the network services
//     // since all network services for potentially multiple devices are colocated on a single thread
//     auto resources = std::make_unique<resources::SystemResources>(
//         system::SystemProvider(tests::make_system([](Options& options) {
//             options.enable_server(true);
//             options.architect_url("localhost:13337");
//             options.placement().resources_strategy(PlacementResources::Dedicated);
//             options.resources().enable_device_memory_pool(true);
//             options.resources().enable_host_memory_pool(true);
//             options.resources().host_memory_pool().block_size(32_MiB);
//             options.resources().host_memory_pool().max_aggregate_bytes(128_MiB);
//             options.resources().device_memory_pool().block_size(64_MiB);
//             options.resources().device_memory_pool().max_aggregate_bytes(128_MiB);
//         })));

//     if (resources->partition_count() < 2 && resources->device_count() < 2)
//     {
//         GTEST_SKIP() << "this test only works with 2 device partitions";
//     }

//     EXPECT_TRUE(resources->partition(0).device());
//     EXPECT_TRUE(resources->partition(1).device());

//     EXPECT_TRUE(resources->partition(0).network());
//     EXPECT_TRUE(resources->partition(1).network());

//     auto h_buffer_0 = resources->partition(0).host().make_buffer(1_MiB);
//     auto d_buffer_0 = resources->partition(0).device()->make_buffer(1_MiB);

//     auto h_ucx_block = resources->partition(0).network()->data_plane().registration_cache().lookup(h_buffer_0.data());
//     auto d_ucx_block = resources->partition(0).network()->data_plane().registration_cache().lookup(d_buffer_0.data());

//     EXPECT_TRUE(h_ucx_block);
//     EXPECT_TRUE(d_ucx_block);

//     EXPECT_EQ(h_ucx_block->bytes(), 32_MiB);
//     EXPECT_EQ(d_ucx_block->bytes(), 64_MiB);

//     EXPECT_TRUE(h_ucx_block->local_handle());
//     EXPECT_TRUE(h_ucx_block->remote_handle());
//     EXPECT_TRUE(h_ucx_block->remote_handle_size());

//     EXPECT_TRUE(d_ucx_block->local_handle());
//     EXPECT_TRUE(d_ucx_block->remote_handle());
//     EXPECT_TRUE(d_ucx_block->remote_handle_size());

//     // the following can not assumed to be true
//     // the remote handle size is proportional to the number and types of ucx transports available in a given domain
//     // EXPECT_LE(h_ucx_block.remote_handle_size(), d_ucx_block.remote_handle_size());

//     // expect that the buffers are allowed to survive pass the resource manager
//     resources.reset();

//     h_buffer_0.release();
//     d_buffer_0.release();
// }

// TEST_F(TestNetwork, CommsSendRecv)
// {
//     // using options.placement().resources_strategy(PlacementResources::Shared)
//     // will test if cudaSetDevice is being properly called by the network services
//     // since all network services for potentially multiple devices are colocated on a single thread
//     auto resources = std::make_unique<resources::SystemResources>(
//         system::SystemProvider(tests::make_system([](Options& options) {
//             options.enable_server(true);
//             options.architect_url("localhost:13337");
//             options.placement().resources_strategy(PlacementResources::Dedicated);
//             options.resources().enable_device_memory_pool(true);
//             options.resources().enable_host_memory_pool(true);
//             options.resources().host_memory_pool().block_size(32_MiB);
//             options.resources().host_memory_pool().max_aggregate_bytes(128_MiB);
//             options.resources().device_memory_pool().block_size(64_MiB);
//             options.resources().device_memory_pool().max_aggregate_bytes(128_MiB);
//         })));

//     if (resources->partition_count() < 2 && resources->device_count() < 2)
//     {
//         GTEST_SKIP() << "this test only works with 2 device partitions";
//     }

//     EXPECT_TRUE(resources->partition(0).network());
//     EXPECT_TRUE(resources->partition(1).network());

//     auto& r0 = resources->partition(0).network()->data_plane();
//     auto& r1 = resources->partition(1).network()->data_plane();

//     // here we are exchanging internal ucx worker addresses without the need of the control plane
//     // r0.client().register_instance(1, r1.ucx_address());  // register r1 as instance_id 1
//     // r1.client().register_instance(0, r0.ucx_address());  // register r0 as instance_id 0

//     // auto f1 = resources->partition(0).network()->control_plane().client().connections().update_future();
//     // auto f2 = resources->partition(1).network()->control_plane().client().connections().update_future();
//     resources->partition(0).network()->control_plane().client().request_update();
//     // f1.get();
//     // f2.get();

//     auto id_0 = resources->partition(0).network()->control_plane().instance_id();
//     auto id_1 = resources->partition(1).network()->control_plane().instance_id();

//     int src = 42;
//     int dst = -1;

//     data_plane::Request send_req;
//     data_plane::Request recv_req;

//     r1.client().async_p2p_recv(&dst, sizeof(int), 0, recv_req);
//     r0.client().async_p2p_send(&src, sizeof(int), 0, id_1, send_req);

//     LOG(INFO) << "await recv";
//     recv_req.await_complete();
//     LOG(INFO) << "await send";
//     send_req.await_complete();

//     EXPECT_EQ(src, dst);

//     // expect that the buffers are allowed to survive pass the resource manager
//     resources.reset();
// }

// TEST_F(TestNetwork, CommsGet)
// {
//     // using options.placement().resources_strategy(PlacementResources::Shared)
//     // will test if cudaSetDevice is being properly called by the network services
//     // since all network services for potentially multiple devices are colocated on a single thread
//     auto resources = std::make_unique<resources::SystemResources>(
//         system::SystemProvider(tests::make_system([](Options& options) {
//             options.enable_server(true);
//             options.architect_url("localhost:13337");
//             options.placement().resources_strategy(PlacementResources::Dedicated);
//             options.resources().enable_device_memory_pool(true);
//             options.resources().enable_host_memory_pool(true);
//             options.resources().host_memory_pool().block_size(32_MiB);
//             options.resources().host_memory_pool().max_aggregate_bytes(128_MiB);
//             options.resources().device_memory_pool().block_size(64_MiB);
//             options.resources().device_memory_pool().max_aggregate_bytes(128_MiB);
//         })));

//     if (resources->partition_count() < 2 && resources->device_count() < 2)
//     {
//         GTEST_SKIP() << "this test only works with 2 device partitions";
//     }

//     EXPECT_TRUE(resources->partition(0).network());
//     EXPECT_TRUE(resources->partition(1).network());

//     auto src = resources->partition(0).host().make_buffer(1_MiB);
//     auto dst = resources->partition(1).host().make_buffer(1_MiB);

//     // here we really want a monad on the optional
//     auto block = resources->partition(0).network()->data_plane().registration_cache().lookup(src.data());
//     EXPECT_TRUE(block);
//     auto src_keys = block->packed_remote_keys();

//     auto* src_data    = static_cast<std::size_t*>(src.data());
//     std::size_t count = 1_MiB / sizeof(std::size_t);
//     for (std::size_t i = 0; i < count; ++i)
//     {
//         src_data[i] = 42;
//     }

//     auto& r0 = resources->partition(0).network()->data_plane();
//     auto& r1 = resources->partition(1).network()->data_plane();

//     // here we are exchanging internal ucx worker addresses without the need of the control plane
//     // auto f1 = resources->partition(0).network()->control_plane().client().connections().update_future();
//     // auto f2 = resources->partition(1).network()->control_plane().client().connections().update_future();
//     resources->partition(0).network()->control_plane().client().request_update();
//     // f1.get();
//     // f2.get();

//     auto id_0 = resources->partition(0).network()->control_plane().instance_id();
//     auto id_1 = resources->partition(1).network()->control_plane().instance_id();

//     data_plane::Request get_req;

//     r1.client().async_get(dst.data(), 1_MiB, id_0, src.data(), src_keys, get_req);

//     LOG(INFO) << "await get";
//     get_req.await_complete();

//     auto* dst_data = static_cast<std::size_t*>(dst.data());
//     for (std::size_t i = 0; i < count; ++i)
//     {
//         EXPECT_EQ(dst_data[i], 42);
//     }

//     // expect that the buffers are allowed to survive pass the resource manager
//     resources.reset();
// }

// TEST_F(TestNetwork, PersistentEagerDataPlaneTaggedRecv)
// {
//     // using options.placement().resources_strategy(PlacementResources::Shared)
//     // will test if cudaSetDevice is being properly called by the network services
//     // since all network services for potentially multiple devices are colocated on a single thread
//     auto resources = std::make_unique<resources::SystemResources>(
//         system::SystemProvider(tests::make_system([](Options& options) {
//             options.enable_server(true);
//             options.architect_url("localhost:13337");
//             options.placement().resources_strategy(PlacementResources::Dedicated);
//             options.resources().enable_device_memory_pool(true);
//             options.resources().enable_host_memory_pool(true);
//             options.resources().host_memory_pool().block_size(32_MiB);
//             options.resources().host_memory_pool().max_aggregate_bytes(128_MiB);
//             options.resources().device_memory_pool().block_size(64_MiB);
//             options.resources().device_memory_pool().max_aggregate_bytes(128_MiB);
//         })));

//     if (resources->partition_count() < 2 && resources->device_count() < 2)
//     {
//         GTEST_SKIP() << "this test only works with 2 device partitions";
//     }

//     // here we are exchanging internal ucx worker addresses without the need of the control plane
//     // auto f1 = resources->partition(0).network()->control_plane().client().connections().update_future();
//     // auto f2 = resources->partition(1).network()->control_plane().client().connections().update_future();
//     resources->partition(0).network()->control_plane().client().request_update();
//     // f1.get();
//     // f2.get();

//     EXPECT_TRUE(resources->partition(0).network());
//     EXPECT_TRUE(resources->partition(1).network());

//     auto& r0 = resources->partition(0).network()->data_plane();
//     auto& r1 = resources->partition(1).network()->data_plane();

//     const std::uint64_t tag          = 20919;
//     std::atomic<std::size_t> counter = 0;

//     auto recv_sink = std::make_unique<node::RxSink<memory::TransientBuffer>>([&](memory::TransientBuffer buffer) {
//         EXPECT_EQ(buffer.bytes(), 128);
//         counter++;
//         // r0.server().deserialize_source().drop_edge(tag);
//     });

//     auto deser_source = r0.server().deserialize_source().get_source(tag);

//     mrc::make_edge(*deser_source, *recv_sink);

//     auto launch_opts = resources->partition(0).network()->data_plane().launch_options(1);
//     auto recv_runner = resources->partition(0)
//                            .runnable()
//                            .launch_control()
//                            .prepare_launcher(launch_opts, std::move(recv_sink))
//                            ->ignition();

//     auto endpoint = r1.client().endpoint_shared(r0.instance_id());

//     data_plane::Request req;
//     auto buffer   = resources->partition(1).host().make_buffer(128);
//     auto send_tag = tag | mrc::data_plane::TAG_EGR_MSG;
//     r1.client().async_send(buffer.data(), buffer.bytes(), send_tag, *endpoint, req);
//     EXPECT_TRUE(req.await_complete());

//     // the channel will be dropped when the first message goes thru
//     recv_runner->await_join();
//     EXPECT_EQ(counter, 1);

//     resources.reset();
// }

TEST_F(TestNetwork, SimpleTaggedMessage)
{
    uint32_t send_data = 42;
    uint32_t recv_data = 0;

    auto receive_request = m_resources->tagged_recv_async(m_loopback_endpoint,
                                                          &recv_data,
                                                          sizeof(uint32_t),
                                                          1,
                                                          data_plane::TagMasks::AnyMsg);

    auto send_request = m_resources->tagged_send_async(m_loopback_endpoint, &send_data, sizeof(uint32_t), 1);

    while (!send_request->isCompleted() || !receive_request->isCompleted())
    {
        m_resources->progress();
    }

    EXPECT_EQ(send_data, recv_data);
}

TEST_F(TestNetwork, SimpleActiveMessage)
{
    uint32_t send_data = 42;
    uint32_t recv_data = 0;

    auto receive_request = m_resources->am_recv_async(m_loopback_endpoint);

    auto send_request =
        m_resources->am_send_async(m_loopback_endpoint, &send_data, sizeof(uint32_t), UCS_MEMORY_TYPE_HOST);

    while (!send_request->isCompleted() || !receive_request->isCompleted())
    {
        m_resources->progress();
    }

    // Now copy the data into the recv_data variable
    std::memcpy(&recv_data, receive_request->getRecvBuffer()->data(), receive_request->getRecvBuffer()->getSize());

    EXPECT_EQ(send_data, recv_data);
}

TEST_F(TestNetwork, LocalDescriptorRoundTrip)
{
    TransferObject send_data = {"test", 42, {1, 2, 3, 4, 5}};

    auto send_data_copy = send_data;

    // Create a descriptor that will pass through the local path
    auto descriptor = runtime::Descriptor2::create_from_value(std::move(send_data_copy), *m_resources);

    // deserialize the descriptor to get value
    auto recv_data = descriptor->deserialize<decltype(send_data)>();

    EXPECT_EQ(send_data, recv_data);
}

TEST_F(TestNetwork, TransferFullDescriptors)
{
    static_assert(codable::member_decodable<ComplexObject>);
    static_assert(codable::member_decodable<TransferObject>);

    ComplexObject send_data = {"test", 42, {"test", 42, std::vector<u_int8_t>(64_KiB)}, std::vector<u_int8_t>(8_KiB)};

    auto send_data_copy = send_data;

    // Create the descriptor object from value
    std::shared_ptr<runtime::Descriptor2> send_descriptor =
        runtime::Descriptor2::create_from_value(std::move(send_data_copy), *m_resources);

    // Check that no remote payloads are yet registered with `DataPlaneResources2`.
    EXPECT_EQ(m_resources->registered_remote_descriptor_count(), 0);

    // Await for registering remote descriptor coroutine to complete
    uint64_t obj_id = coroutines::sync_wait(m_resources->register_remote_descriptor(send_descriptor));

    // Get the serialized data
    auto serialized_data           = send_descriptor->serialize(memory::malloc_memory_resource::instance());
    auto send_descriptor_object_id = send_descriptor->encoded_object().object_id();

    // Check that there is exactly 1 registered descriptor
    EXPECT_EQ(m_resources->registered_remote_descriptor_count(), 1);
    EXPECT_EQ(m_resources->registered_remote_descriptor_ptr_count(send_descriptor_object_id), 1);

    send_descriptor = nullptr;

    auto receive_request = m_resources->am_recv_async(m_loopback_endpoint);
    auto send_request    = m_resources->am_send_async(m_loopback_endpoint, serialized_data);

    while (!send_request->isCompleted() || !receive_request->isCompleted())
    {
        m_resources->progress();
    }

    // Acquire the registered descriptor as a `weak_ptr` which we can use to immediately verify to be valid, but
    // invalid once `DataPlaneResources2` releases it.
    std::weak_ptr<runtime::Descriptor2> registered_send_descriptor = m_resources->get_descriptor(send_descriptor_object_id);
    EXPECT_NE(registered_send_descriptor.lock(), nullptr);

    // Create a descriptor from the received data
    auto buffer_view = memory::buffer_view(receive_request->getRecvBuffer()->data(),
                                           receive_request->getRecvBuffer()->getSize(),
                                           mrc::memory::memory_kind::host);

    // Create the descriptor object from received data
    std::shared_ptr<runtime::Descriptor2> recv_descriptor =
        runtime::Descriptor2::create_from_bytes(std::move(buffer_view), *m_resources);

    // Pull the remaining deferred payloads from the remote machine
    recv_descriptor->fetch_remote_payloads();

    uint64_t recv_descriptor_object_id = recv_descriptor->encoded_object().object_id();

    EXPECT_EQ(send_descriptor_object_id, recv_descriptor_object_id);

    // Wait for remote decrement messages.
    while (registered_send_descriptor.lock() != nullptr)
        m_resources->progress();

    // Redundant with the above, but clarify intent.
    EXPECT_EQ(registered_send_descriptor.lock(), nullptr);

    // Check all remote payloads have been deregistered, including the one previously transferred.
    EXPECT_EQ(m_resources->registered_remote_descriptor_count(), 0);
    EXPECT_THROW(m_resources->registered_remote_descriptor_ptr_count(send_descriptor_object_id), std::out_of_range);

    // Finally, get the value
    auto recv_data = recv_descriptor->deserialize<decltype(send_data)>();

    EXPECT_EQ(send_data, recv_data);
}

TEST_F(TestNetwork, TransferFullDescriptorsDevice)
{
    static_assert(codable::member_decodable<ComplexObject>);
    static_assert(codable::member_decodable<TransferObject>);

    const size_t data_size = 64_KiB;
    std::vector<u_int8_t> send_data_host(data_size);

    u_int8_t* send_data_device;
    cudaMalloc(&send_data_device, send_data_host.size() * sizeof(u_int8_t));
    cudaMemcpy(send_data_device, send_data_host.data(), send_data_host.size() * sizeof(u_int8_t), cudaMemcpyHostToDevice);

    // Create the descriptor object from value
    std::shared_ptr<runtime::Descriptor2> send_descriptor =
        runtime::Descriptor2::create_from_value(std::move(send_data_device), *m_resources);

    // Check that no remote payloads are yet registered with `DataPlaneResources2`.
    EXPECT_EQ(m_resources->registered_remote_descriptor_count(), 0);

    // Await for registering remote descriptor coroutine to complete
    uint64_t obj_id = coroutines::sync_wait(m_resources->register_remote_descriptor(send_descriptor));

    // Get the serialized data
    auto serialized_data           = send_descriptor->serialize(memory::malloc_memory_resource::instance());
    auto send_descriptor_object_id = send_descriptor->encoded_object().object_id();

    // Check that there is exactly 1 registered descriptor
    EXPECT_EQ(m_resources->registered_remote_descriptor_count(), 1);
    EXPECT_EQ(m_resources->registered_remote_descriptor_ptr_count(send_descriptor_object_id), 1);

    send_descriptor = nullptr;

    auto receive_request = m_resources->am_recv_async(m_loopback_endpoint);
    auto send_request    = m_resources->am_send_async(m_loopback_endpoint, serialized_data);

    while (!send_request->isCompleted() || !receive_request->isCompleted())
    {
        m_resources->progress();
    }

    // Acquire the registered descriptor as a `weak_ptr` which we can use to immediately verify to be valid, but
    // invalid once `DataPlaneResources2` releases it.
    std::weak_ptr<runtime::Descriptor2> registered_send_descriptor = m_resources->get_descriptor(send_descriptor_object_id);
    EXPECT_NE(registered_send_descriptor.lock(), nullptr);

    auto buffer_view = memory::buffer_view(receive_request->getRecvBuffer()->data(),
                                           receive_request->getRecvBuffer()->getSize(),
                                           mrc::memory::memory_kind::host);

    // Create the descriptor object from received data
    std::shared_ptr<runtime::Descriptor2> recv_descriptor =
        runtime::Descriptor2::create_from_bytes(std::move(buffer_view), *m_resources);

    // Pull the remaining deferred payloads from the remote machine
    recv_descriptor->fetch_remote_payloads();

    uint64_t recv_descriptor_object_id = recv_descriptor->encoded_object().object_id();

    EXPECT_EQ(send_descriptor_object_id, recv_descriptor_object_id);

    // Wait for remote decrement messages.
    while (registered_send_descriptor.lock() != nullptr)
        m_resources->progress();

    // Redundant with the above, but clarify intent.
    EXPECT_EQ(registered_send_descriptor.lock(), nullptr);

    // Check all remote payloads have been deregistered, including the one previously transferred.
    EXPECT_EQ(m_resources->registered_remote_descriptor_count(), 0);
    EXPECT_THROW(m_resources->registered_remote_descriptor_ptr_count(send_descriptor_object_id), std::out_of_range);

    // Finally, get the value
    auto recv_data_device = recv_descriptor->deserialize<decltype(send_data_device)>();

    // Copy the data into host memory to easily compare the results
    std::vector<u_int8_t> recv_data_host(data_size);
    cudaMemcpy(recv_data_host.data(), recv_data_device, data_size * sizeof(u_int8_t), cudaMemcpyDeviceToHost);

    EXPECT_EQ(send_data_host, recv_data_host);

    // Free device memory
    cudaFree(send_data_device);
    cudaFree(recv_data_device);
}

TEST_F(TestNetwork, TransferFullDescriptorsBroadcast)
{
    // Create resources to simulate remote processes
    auto resources_recv1 = std::make_unique<DataPlaneResources2Tester>();
    resources_recv1->set_instance_id(43);
    auto resources_recv2 = std::make_unique<DataPlaneResources2Tester>();
    resources_recv2->set_instance_id(44);

    auto endpoint_recv1 = m_resources->create_endpoint(resources_recv1->address(), resources_recv1->get_instance_id());
    auto endpoint_recv2 = m_resources->create_endpoint(resources_recv2->address(), resources_recv2->get_instance_id());

    auto endpoint_send1 = resources_recv1->create_endpoint(m_resources->address(), m_resources->get_instance_id());
    auto endpoint_send2 = resources_recv2->create_endpoint(m_resources->address(), m_resources->get_instance_id());

    // Create initial data
    static_assert(codable::decodable<TransferObject>);

    TransferObject send_data = {"test", 42, std::vector<u_int8_t>(64_KiB)};

    auto send_data_copy = send_data;

    // Create the descriptor object from value
    std::shared_ptr<runtime::Descriptor2> send_descriptor =
        runtime::Descriptor2::create_from_value(std::move(send_data_copy), *m_resources);

    // Check that no remote payloads are yet registered with `DataPlaneResources2`.
    EXPECT_EQ(m_resources->registered_remote_descriptor_count(), 0);

    // Await for registering remote descriptor coroutines to complete
    uint64_t obj_id1 = coroutines::sync_wait(m_resources->register_remote_descriptor(send_descriptor));
    uint64_t obj_id2 = coroutines::sync_wait(m_resources->register_remote_descriptor(send_descriptor));

    auto serialized_data1 = send_descriptor->serialize(memory::malloc_memory_resource::instance());
    auto serialized_data2 = send_descriptor->serialize(memory::malloc_memory_resource::instance());
    auto send_descriptor_object_id = send_descriptor->encoded_object().object_id();

    // Check that there is exactly 1 registered descriptor but there are 2 pointers to the same descriptor
    EXPECT_EQ(m_resources->registered_remote_descriptor_count(), 1);
    EXPECT_EQ(m_resources->registered_remote_descriptor_ptr_count(send_descriptor_object_id), 2);

    send_descriptor = nullptr;

    auto processRequest = [this, &send_data, send_descriptor_object_id](auto& resources_recv,
                                                                        auto& endpoint_recv,
                                                                        auto& endpoint_send,
                                                                        auto& serialized_data,
                                                                        auto expected_ptrs) {
        auto receive_request = resources_recv->am_recv_async(endpoint_send);
        auto send_request    = m_resources->am_send_async(endpoint_recv, serialized_data);

        while (!send_request->isCompleted() || !receive_request->isCompleted())
        {
            m_resources->progress();
            resources_recv->progress();
        }

        // Acquire the registered descriptor as a `weak_ptr` which we can use to immediately verify to be valid, but
        // invalid once `DataPlaneResources2` releases it.
        std::weak_ptr<runtime::Descriptor2> registered_send_descriptor = m_resources->get_descriptor(send_descriptor_object_id);
        EXPECT_NE(registered_send_descriptor.lock(), nullptr);

        auto buffer_view = memory::buffer_view(receive_request->getRecvBuffer()->data(),
                                              receive_request->getRecvBuffer()->getSize(),
                                              mrc::memory::memory_kind::host);

        // Create the descriptor object from received data
        std::shared_ptr<runtime::Descriptor2> recv_descriptor =
            runtime::Descriptor2::create_from_bytes(std::move(buffer_view), *m_resources);

        // Pull the remaining deferred payloads from the remote machine
        recv_descriptor->fetch_remote_payloads();

        uint64_t recv_descriptor_object_id = recv_descriptor->encoded_object().object_id();

        if (expected_ptrs > 0)
        {
            // Wait for remote decrement messages.
            while (m_resources->registered_remote_descriptor_ptr_count(send_descriptor_object_id) != expected_ptrs)
            {
                m_resources->progress();
                resources_recv->progress();
            }

            // Redundant with the above, but clarify intent.
            EXPECT_EQ(m_resources->registered_remote_descriptor_ptr_count(send_descriptor_object_id), expected_ptrs);
        }
        else
        {
            // Wait for remote decrement messages.
            while (registered_send_descriptor.lock() != nullptr)
            {
                m_resources->progress();
                resources_recv->progress();
            }

            // Redundant with the above, but clarify intent.
            EXPECT_EQ(registered_send_descriptor.lock(), nullptr);
        }

        // Finally, get the value
        auto recv_data = recv_descriptor->deserialize<decltype(send_data)>();

        EXPECT_EQ(send_data, recv_data);
    };

    processRequest(resources_recv1, endpoint_recv1, endpoint_send1, serialized_data1, 1);
    processRequest(resources_recv2, endpoint_recv2, endpoint_send2, serialized_data2, 0);
}

class TestNetworkPressure : public TestNetwork, public ::testing::WithParamInterface<bool>
{};

TEST_P(TestNetworkPressure, TransferPressureControl)
{
    static_assert(codable::decodable<TransferObject>);

    auto block_provider = std::make_shared<memory::memory_block_provider>();

    TransferObject send_data = {"test", 42, std::vector<u_int8_t>(64_KiB)};

    size_t max_descriptors{3};
    size_t total_descriptors{10};
    size_t registered_descriptors{0};
    m_resources->set_max_remote_descriptors(max_descriptors);

    bool registration_finished{false};
    std::queue<memory::buffer> serialized_data;
    std::queue<uint64_t> send_descriptor_object_ids;

    auto register_descriptors = [this,
                                 block_provider,
                                 &send_data,
                                 &serialized_data,
                                 &send_descriptor_object_ids,
                                 &registered_descriptors,
                                 total_descriptors,
                                 &registration_finished]() {
        for (size_t i = 0; i < total_descriptors; ++i)
        {
            auto send_data_copy = send_data;

            // Create the descriptor object from received data
            std::shared_ptr<runtime::Descriptor2> send_descriptor =
                runtime::Descriptor2::create_from_value(std::move(send_data_copy), *m_resources);

            // Await for registering remote descriptor coroutines to complete
            uint64_t obj_id = coroutines::sync_wait(m_resources->register_remote_descriptor(send_descriptor));

            // Get the serialized data and push to queue for consumption by request processing thread
            serialized_data.push(send_descriptor->serialize(memory::malloc_memory_resource::instance()));
            send_descriptor_object_ids.push(send_descriptor->encoded_object().object_id());

            send_descriptor = nullptr;

            ++registered_descriptors;
        }
        registration_finished = true;
    };

    auto increase_max_descriptors =
        [this, max_descriptors, total_descriptors, &registered_descriptors]() {
            // Wait until registration hits max number of descriptors and blocks
            while (registered_descriptors < max_descriptors) {}

            // Unblock `register_descriptors` immediately, even if requests are not being processed
            m_resources->set_max_remote_descriptors(total_descriptors);
        };

    auto process_request = [this,
                            &send_data,
                            block_provider,
                            max_descriptors,
                            &registration_finished,
                            &serialized_data,
                            &send_descriptor_object_ids](uint64_t index) {
        // Block processing requests until either the maximum number of remote descriptors is registered
        // (`DataPlaneResources2` internal queue is full) and registrations are still ongoing or serialized data is not
        // available yet.
        while ((m_resources->registered_remote_descriptor_count() < max_descriptors && !registration_finished) ||
               serialized_data.empty()) {}

        auto local_serialized_data = std::move(serialized_data.front());
        serialized_data.pop();

        auto receive_request = m_resources->am_recv_async(m_loopback_endpoint);
        auto send_request = m_resources->am_send_async(m_loopback_endpoint, local_serialized_data);

        while (!send_request->isCompleted() || !receive_request->isCompleted())
        {
            m_resources->progress();
        }

        // Acquire the registered descriptor as a `weak_ptr` which we can use to immediately verify to be valid, but
        // invalid once `DataPlaneResources2` releases it.
        auto send_descriptor_object_id = send_descriptor_object_ids.front();
        send_descriptor_object_ids.pop();

        std::weak_ptr<mrc::runtime::Descriptor2> registered_send_descriptor = m_resources->get_descriptor(
            send_descriptor_object_id);

        EXPECT_NE(registered_send_descriptor.lock(), nullptr);

        // Create the descriptor object from received data
        auto recv_descriptor = runtime::Descriptor2::create_from_bytes(
            {receive_request->getRecvBuffer()->data(),
             receive_request->getRecvBuffer()->getSize(),
             mrc::memory::memory_kind::host},
            *m_resources);

        // Pull the remaining deferred payloads from the remote machine
        recv_descriptor->fetch_remote_payloads();

        auto recv_descriptor_object_id = recv_descriptor->encoded_object().object_id();

        EXPECT_EQ(send_descriptor_object_id, recv_descriptor_object_id);

        // TODO(Peter): This is now completely async and we must progress the worker, we need a timeout in case it
        // fails to complete. Wait for remote decrement messages.
        while (registered_send_descriptor.lock() != nullptr)
        {
            m_resources->progress();
        }

        // Redundant with the above, but clarify intent.
        EXPECT_EQ(registered_send_descriptor.lock(), nullptr);

        // Finally, get the value
        auto recv_data = recv_descriptor->deserialize<decltype(send_data)>();

        EXPECT_EQ(send_data, recv_data);
    };

    if (GetParam())
    {
        std::thread increase_max_thread(increase_max_descriptors);
        increase_max_thread.join();
    }

    // Launch thread to register remote descriptors, which will block once max_remote_descriptors are pending
    std::thread process_thread(register_descriptors);

    // Process requests in current thread
    for (size_t i = 0; i < total_descriptors; ++i)
    {
        process_request(i);
    }

    process_thread.join();
    EXPECT_EQ(m_resources->registered_remote_descriptor_count(), 0);
}

INSTANTIATE_TEST_SUITE_P(IncreaseMaxRemoteDescriptorsWhileRunning, TestNetworkPressure, ::testing::Values(false, true));

// TEST_F(TestNetwork, NetworkEventsManagerLifeCycle)
// {
//     auto launcher = m_launch_control->prepare_launcher(std::move(m_mutable_nem));

//     // auto& service = m_launch_control->service(runnable::MrcService::data_plane::Server);
//     // service.stop();
//     // service.await_join();
// }

// TEST_F(TestNetwork, data_plane::Server)
// {
//     GTEST_SKIP() << "blocked by #121";

//     std::atomic<std::size_t> counter_0 = 0;
//     std::atomic<std::size_t> counter_1 = 0;

//     boost::fibers::barrier barrier(2);

//     // create a deserialize sink which access a memory::block
//     auto sink_0 = std::make_unique<node::RxSink<memory::block>>();
//     sink_0->set_observer([&counter_0, &barrier](memory::block block) {
//         std::free(block.data());
//         ++counter_0;
//         barrier.wait();
//     });

//     auto sink_1 = std::make_unique<node::RxSink<memory::block>>();
//     sink_1->set_observer([&counter_1, &barrier](memory::block block) {
//         std::free(block.data());
//         ++counter_1;
//         barrier.wait();
//     });

//     mrc::make_edge(m_mutable_nem->deserialize_source().source(0), *sink_0);
//     mrc::make_edge(m_mutable_nem->deserialize_source().source(1), *sink_1);

//     auto nem_worker_address = m_mutable_nem->worker_address();

//     auto runner_0 = m_launch_control->prepare_launcher(std::move(sink_0))->ignition();
//     auto runner_1 = m_launch_control->prepare_launcher(std::move(sink_1))->ignition();
//     auto service  = m_launch_control->prepare_launcher(std::move(m_mutable_nem))->ignition();

//     // NEM is running along with two Sinks attached to the deserialization router
//     auto comm = std::make_unique<data_plane::Client>(std::make_shared<ucx::Worker>(m_context),
//     *m_launch_control); comm->register_instance(0, nem_worker_address);

//     double val = 3.14;
//     codable::EncodedObject encoded_val;
//     codable::encode(val, encoded_val);

//     // send a ucx tagged message with a memory block of a fixed sized with a destination port address of 0
//     comm->await_send(0, 0, encoded_val);
//     barrier.wait();
//     EXPECT_EQ(counter_0, 1);
//     EXPECT_EQ(counter_1, 0);

//     // send a ucx tagged message with a memory block of a fixed sized with a destination port address of 1
//     comm->await_send(0, 1, encoded_val);
//     barrier.wait();
//     EXPECT_EQ(counter_0, 1);
//     EXPECT_EQ(counter_1, 1);

//     service->stop();
//     service->await_join();

//     runner_0->await_join();
//     runner_1->await_join();

//     comm.reset();
// }

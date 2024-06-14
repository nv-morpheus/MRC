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

#include "mrc/channel/status.hpp"
#include "mrc/codable/decode.hpp"
#include "mrc/codable/fundamental_types.hpp"  // IWYU pragma: keep
#include "mrc/codable/type_traits.hpp"
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

#include <boost/fiber/fiber.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <rxcpp/rx.hpp>
#include <sys/types.h>
#include <ucs/memory/memory_type.h>
#include <ucxx/api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <stop_token>
#include <utility>

using namespace mrc;
using namespace mrc::memory::literals;

class DataPlaneResources2Tester : public data_plane::DataPlaneResources2
{
  public:
    std::shared_ptr<runtime::RemoteDescriptorImpl2> get_descriptor(uint64_t object_id) override
    {
        return m_remote_descriptor_by_id[object_id];
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

namespace mrc::codable {

// template <typename T>
// struct codable_protocol<std::vector<T>, std::enable_if_t<std::is_fundamental_v<T>>>
// {
//     static void serialize(const std::vector<T>& obj,
//                           mrc::codable::Encoder2<std::vector<T>>& encoder,
//                           const mrc::codable::EncodingOptions& opts = {})
//     {
//         // First put in the size
//         mrc::codable::encode2(obj.size(), encoder, opts);

//         // Since these are fundamental types, just encode in a single memory block
//         encoder.register_memory_view({obj.data(), obj.size() * sizeof(T), memory::memory_kind::host});
//     }

//     static void serialize(const std::vector<T>& obj,
//                           mrc::codable::Encoder<std::vector<T>>& encoder,
//                           const mrc::codable::EncodingOptions& opts = {})
//     {
//         // First put in the size
//         mrc::codable::encode2(obj.size(), encoder, opts);

//         // Since these are fundamental types, just encode in a single memory block
//         encoder.register_memory_view({obj.data(), obj.size() * sizeof(T), memory::memory_kind::host});
//     }

//     static std::vector<T> decode(const Decoder<T>& encoding, std::size_t object_idx)
//     {
//         // Get the first item to get the size
//         auto count = mrc::codable::decode<size_t>(encoding, object_idx);

//         std::vector<T> object = std::vector<T>(count);

//         // encoding.deserialize(std::size_t object_idx)
//         return object;
//     }
// };

}  // namespace mrc::codable

class TransferObject
{
  public:
    TransferObject() = default;
    TransferObject(std::string name, int value, std::vector<int> data) :
      m_has_data(true),
      m_name(std::move(name)),
      m_value(value),
      m_data(std::move(data))
    {}

    TransferObject(const TransferObject& other)            = default;
    TransferObject& operator=(const TransferObject& other) = default;

    // Move constructor
    TransferObject(TransferObject&& other) noexcept :
      m_has_data(std::exchange(other.m_has_data, false)),
      m_name(std::move(other.m_name)),
      m_value(std::exchange(other.m_value, 0)),
      m_data(std::move(other.m_data))
    {}

    // Move assignment
    TransferObject& operator=(TransferObject&& other) noexcept
    {
        if (this != &other)
        {
            m_has_data = std::exchange(other.m_has_data, false);
            m_name     = std::move(other.m_name);
            m_value    = std::exchange(other.m_value, 0);
            m_data     = std::move(other.m_data);
        }
        return *this;
    }

    ~TransferObject()
    {
        if (m_has_data)
        {
            LOG(INFO) << "TransferObject dtor when it has data";
        }
    }

    // int a() const { return m_a; }
    // int b() const { return m_b; }

    // void set_a(int a) { m_a = a; }
    // void set_b(int b) { m_b = b; }

    bool operator==(const TransferObject& other) const
    {
        return m_name == other.m_name && m_value == other.m_value && m_data == other.m_data;
    }

    void serialize(mrc::codable::Encoder<TransferObject>& encoder, const mrc::codable::EncodingOptions& opts) const {}

    void serialize(mrc::codable::Encoder2<TransferObject>& encoder, const mrc::codable::EncodingOptions& opts) const
    {
        mrc::codable::encode2(m_name, encoder, opts);
        mrc::codable::encode2(m_value, encoder, opts);
        mrc::codable::encode2(m_data, encoder, opts);
    }

    static TransferObject deserialize(const mrc::codable::Decoder<TransferObject>& decoder, std::size_t object_idx)
    {
        TransferObject obj;
        // mrc::codable::decode2(decoder, obj.m_value);
        // mrc::codable::decode2(decoder, obj.m_data);
        return obj;
    }

    static TransferObject deserialize(const mrc::codable::Decoder2<TransferObject>& decoder, size_t object_idx)
    {
        TransferObject obj;

        obj.m_has_data = true;
        obj.m_name     = mrc::codable::decode2<std::string, TransferObject>(decoder, object_idx);
        obj.m_value    = mrc::codable::decode2<int, TransferObject>(decoder, object_idx);
        obj.m_data     = mrc::codable::decode2<std::vector<int>, TransferObject>(decoder, object_idx);

        return obj;
    }

  private:
    bool m_has_data{false};

    std::string m_name;
    int m_value{0};

    std::vector<int> m_data;
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

TEST_F(TestNetwork, ResourceManager)
{
    // using options.placement().resources_strategy(PlacementResources::Shared)
    // will test if cudaSetDevice is being properly called by the network services
    // since all network services for potentially multiple devices are colocated on a single thread
    auto resources = std::make_unique<resources::SystemResources>(
        system::SystemProvider(tests::make_system([](Options& options) {
            options.enable_server(true);
            options.architect_url("localhost:13337");
            options.placement().resources_strategy(PlacementResources::Dedicated);
            options.resources().enable_device_memory_pool(true);
            options.resources().enable_host_memory_pool(true);
            options.resources().host_memory_pool().block_size(32_MiB);
            options.resources().host_memory_pool().max_aggregate_bytes(128_MiB);
            options.resources().device_memory_pool().block_size(64_MiB);
            options.resources().device_memory_pool().max_aggregate_bytes(128_MiB);
        })));

    if (resources->partition_count() < 2 && resources->device_count() < 2)
    {
        GTEST_SKIP() << "this test only works with 2 device partitions";
    }

    EXPECT_TRUE(resources->partition(0).device());
    EXPECT_TRUE(resources->partition(1).device());

    EXPECT_TRUE(resources->partition(0).network());
    EXPECT_TRUE(resources->partition(1).network());

    auto h_buffer_0 = resources->partition(0).host().make_buffer(1_MiB);
    auto d_buffer_0 = resources->partition(0).device()->make_buffer(1_MiB);

    auto h_ucx_block = resources->partition(0).network()->data_plane().registration_cache().lookup(h_buffer_0.data());
    auto d_ucx_block = resources->partition(0).network()->data_plane().registration_cache().lookup(d_buffer_0.data());

    EXPECT_TRUE(h_ucx_block);
    EXPECT_TRUE(d_ucx_block);

    EXPECT_EQ(h_ucx_block->bytes(), 32_MiB);
    EXPECT_EQ(d_ucx_block->bytes(), 64_MiB);

    EXPECT_TRUE(h_ucx_block->local_handle());
    EXPECT_TRUE(h_ucx_block->remote_handle());
    EXPECT_TRUE(h_ucx_block->remote_handle_size());

    EXPECT_TRUE(d_ucx_block->local_handle());
    EXPECT_TRUE(d_ucx_block->remote_handle());
    EXPECT_TRUE(d_ucx_block->remote_handle_size());

    // the following can not assumed to be true
    // the remote handle size is proportional to the number and types of ucx transports available in a given domain
    // EXPECT_LE(h_ucx_block.remote_handle_size(), d_ucx_block.remote_handle_size());

    // expect that the buffers are allowed to survive pass the resource manager
    resources.reset();

    h_buffer_0.release();
    d_buffer_0.release();
}

TEST_F(TestNetwork, CommsSendRecv)
{
    // using options.placement().resources_strategy(PlacementResources::Shared)
    // will test if cudaSetDevice is being properly called by the network services
    // since all network services for potentially multiple devices are colocated on a single thread
    auto resources = std::make_unique<resources::SystemResources>(
        system::SystemProvider(tests::make_system([](Options& options) {
            options.enable_server(true);
            options.architect_url("localhost:13337");
            options.placement().resources_strategy(PlacementResources::Dedicated);
            options.resources().enable_device_memory_pool(true);
            options.resources().enable_host_memory_pool(true);
            options.resources().host_memory_pool().block_size(32_MiB);
            options.resources().host_memory_pool().max_aggregate_bytes(128_MiB);
            options.resources().device_memory_pool().block_size(64_MiB);
            options.resources().device_memory_pool().max_aggregate_bytes(128_MiB);
        })));

    if (resources->partition_count() < 2 && resources->device_count() < 2)
    {
        GTEST_SKIP() << "this test only works with 2 device partitions";
    }

    EXPECT_TRUE(resources->partition(0).network());
    EXPECT_TRUE(resources->partition(1).network());

    auto& r0 = resources->partition(0).network()->data_plane();
    auto& r1 = resources->partition(1).network()->data_plane();

    // here we are exchanging internal ucx worker addresses without the need of the control plane
    // r0.client().register_instance(1, r1.ucx_address());  // register r1 as instance_id 1
    // r1.client().register_instance(0, r0.ucx_address());  // register r0 as instance_id 0

    // auto f1 = resources->partition(0).network()->control_plane().client().connections().update_future();
    // auto f2 = resources->partition(1).network()->control_plane().client().connections().update_future();
    resources->partition(0).network()->control_plane().client().request_update();
    // f1.get();
    // f2.get();

    auto id_0 = resources->partition(0).network()->control_plane().instance_id();
    auto id_1 = resources->partition(1).network()->control_plane().instance_id();

    int src = 42;
    int dst = -1;

    data_plane::Request send_req;
    data_plane::Request recv_req;

    r1.client().async_p2p_recv(&dst, sizeof(int), 0, recv_req);
    r0.client().async_p2p_send(&src, sizeof(int), 0, id_1, send_req);

    LOG(INFO) << "await recv";
    recv_req.await_complete();
    LOG(INFO) << "await send";
    send_req.await_complete();

    EXPECT_EQ(src, dst);

    // expect that the buffers are allowed to survive pass the resource manager
    resources.reset();
}

TEST_F(TestNetwork, CommsGet)
{
    // using options.placement().resources_strategy(PlacementResources::Shared)
    // will test if cudaSetDevice is being properly called by the network services
    // since all network services for potentially multiple devices are colocated on a single thread
    auto resources = std::make_unique<resources::SystemResources>(
        system::SystemProvider(tests::make_system([](Options& options) {
            options.enable_server(true);
            options.architect_url("localhost:13337");
            options.placement().resources_strategy(PlacementResources::Dedicated);
            options.resources().enable_device_memory_pool(true);
            options.resources().enable_host_memory_pool(true);
            options.resources().host_memory_pool().block_size(32_MiB);
            options.resources().host_memory_pool().max_aggregate_bytes(128_MiB);
            options.resources().device_memory_pool().block_size(64_MiB);
            options.resources().device_memory_pool().max_aggregate_bytes(128_MiB);
        })));

    if (resources->partition_count() < 2 && resources->device_count() < 2)
    {
        GTEST_SKIP() << "this test only works with 2 device partitions";
    }

    EXPECT_TRUE(resources->partition(0).network());
    EXPECT_TRUE(resources->partition(1).network());

    auto src = resources->partition(0).host().make_buffer(1_MiB);
    auto dst = resources->partition(1).host().make_buffer(1_MiB);

    // here we really want a monad on the optional
    auto block = resources->partition(0).network()->data_plane().registration_cache().lookup(src.data());
    EXPECT_TRUE(block);
    auto src_keys = block->packed_remote_keys();

    auto* src_data    = static_cast<std::size_t*>(src.data());
    std::size_t count = 1_MiB / sizeof(std::size_t);
    for (std::size_t i = 0; i < count; ++i)
    {
        src_data[i] = 42;
    }

    auto& r0 = resources->partition(0).network()->data_plane();
    auto& r1 = resources->partition(1).network()->data_plane();

    // here we are exchanging internal ucx worker addresses without the need of the control plane
    // auto f1 = resources->partition(0).network()->control_plane().client().connections().update_future();
    // auto f2 = resources->partition(1).network()->control_plane().client().connections().update_future();
    resources->partition(0).network()->control_plane().client().request_update();
    // f1.get();
    // f2.get();

    auto id_0 = resources->partition(0).network()->control_plane().instance_id();
    auto id_1 = resources->partition(1).network()->control_plane().instance_id();

    data_plane::Request get_req;

    r1.client().async_get(dst.data(), 1_MiB, id_0, src.data(), src_keys, get_req);

    LOG(INFO) << "await get";
    get_req.await_complete();

    auto* dst_data = static_cast<std::size_t*>(dst.data());
    for (std::size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(dst_data[i], 42);
    }

    // expect that the buffers are allowed to survive pass the resource manager
    resources.reset();
}

TEST_F(TestNetwork, PersistentEagerDataPlaneTaggedRecv)
{
    // using options.placement().resources_strategy(PlacementResources::Shared)
    // will test if cudaSetDevice is being properly called by the network services
    // since all network services for potentially multiple devices are colocated on a single thread
    auto resources = std::make_unique<resources::SystemResources>(
        system::SystemProvider(tests::make_system([](Options& options) {
            options.enable_server(true);
            options.architect_url("localhost:13337");
            options.placement().resources_strategy(PlacementResources::Dedicated);
            options.resources().enable_device_memory_pool(true);
            options.resources().enable_host_memory_pool(true);
            options.resources().host_memory_pool().block_size(32_MiB);
            options.resources().host_memory_pool().max_aggregate_bytes(128_MiB);
            options.resources().device_memory_pool().block_size(64_MiB);
            options.resources().device_memory_pool().max_aggregate_bytes(128_MiB);
        })));

    if (resources->partition_count() < 2 && resources->device_count() < 2)
    {
        GTEST_SKIP() << "this test only works with 2 device partitions";
    }

    // here we are exchanging internal ucx worker addresses without the need of the control plane
    // auto f1 = resources->partition(0).network()->control_plane().client().connections().update_future();
    // auto f2 = resources->partition(1).network()->control_plane().client().connections().update_future();
    resources->partition(0).network()->control_plane().client().request_update();
    // f1.get();
    // f2.get();

    EXPECT_TRUE(resources->partition(0).network());
    EXPECT_TRUE(resources->partition(1).network());

    auto& r0 = resources->partition(0).network()->data_plane();
    auto& r1 = resources->partition(1).network()->data_plane();

    const std::uint64_t tag          = 20919;
    std::atomic<std::size_t> counter = 0;

    auto recv_sink = std::make_unique<node::RxSink<memory::TransientBuffer>>([&](memory::TransientBuffer buffer) {
        EXPECT_EQ(buffer.bytes(), 128);
        counter++;
        // r0.server().deserialize_source().drop_edge(tag);
    });

    auto deser_source = r0.server().deserialize_source().get_source(tag);

    mrc::make_edge(*deser_source, *recv_sink);

    auto launch_opts = resources->partition(0).network()->data_plane().launch_options(1);
    auto recv_runner = resources->partition(0)
                           .runnable()
                           .launch_control()
                           .prepare_launcher(launch_opts, std::move(recv_sink))
                           ->ignition();

    auto endpoint = r1.client().endpoint_shared(r0.instance_id());

    data_plane::Request req;
    auto buffer   = resources->partition(1).host().make_buffer(128);
    auto send_tag = tag | mrc::data_plane::TAG_EGR_MSG;
    r1.client().async_send(buffer.data(), buffer.bytes(), send_tag, *endpoint, req);
    EXPECT_TRUE(req.await_complete());

    // the channel will be dropped when the first message goes thru
    recv_runner->await_join();
    EXPECT_EQ(counter, 1);

    resources.reset();
}

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

TEST_F(TestNetwork, TransferStorageObject)
{
    auto send_encoded_obj = std::make_unique<codable::LocalSerializedWrapper>();

    uint32_t send_data = 42;

    // Add some data to the stored object
    send_encoded_obj->add_eager_descriptor({&send_data, sizeof(uint32_t), memory::memory_kind::host});

    // Get the serialized data
    auto serialized_data = send_encoded_obj->to_bytes(memory::malloc_memory_resource::instance());

    auto receive_request = m_resources->am_recv_async(m_loopback_endpoint);

    auto send_request = m_resources->am_send_async(m_loopback_endpoint, serialized_data);

    while (!send_request->isCompleted() || !receive_request->isCompleted())
    {
        m_resources->progress();
    }

    auto recv_encoded_proto = codable::LocalSerializedWrapper::from_bytes({receive_request->getRecvBuffer()->data(),
                                                                           receive_request->getRecvBuffer()->getSize(),
                                                                           mrc::memory::memory_kind::host});

    EXPECT_EQ(*send_encoded_obj, *recv_encoded_proto);
}

TEST_F(TestNetwork, TransferEncodedObjectViaEncode)
{
    auto block_provider = std::make_shared<memory::memory_block_provider>();

    uint32_t send_data = 42;

    // Create the encoded object by encoding the data
    auto send_encoded_obj = codable::encode2(send_data, block_provider);

    // Get the serialized data
    auto serialized_data = send_encoded_obj->to_bytes(memory::malloc_memory_resource::instance());

    auto receive_request = m_resources->am_recv_async(m_loopback_endpoint);

    auto send_request = m_resources->am_send_async(m_loopback_endpoint, serialized_data);

    while (!send_request->isCompleted() || !receive_request->isCompleted())
    {
        m_resources->progress();
    }

    auto recv_encoded_proto = codable::LocalSerializedWrapper::from_bytes({receive_request->getRecvBuffer()->data(),
                                                                           receive_request->getRecvBuffer()->getSize(),
                                                                           mrc::memory::memory_kind::host});

    EXPECT_EQ(*send_encoded_obj, *recv_encoded_proto);
}

TEST_F(TestNetwork, LocalDescriptorRoundTrip)
{
    auto block_provider = std::make_shared<memory::memory_block_provider>();

    TransferObject send_data = {"test", 42, {1, 2, 3, 4, 5}};

    auto send_data_copy = send_data;

    // Create a value descriptor
    auto send_value_descriptor = runtime::TypedValueDescriptor<decltype(send_data)>::create(std::move(send_data_copy));

    // Convert to a local descriptor
    auto send_local_descriptor = runtime::LocalDescriptor2::from_value(std::move(send_value_descriptor),
                                                                       block_provider);

    auto recv_value_descriptor = runtime::TypedValueDescriptor<decltype(send_data)>::from_local(
        std::move(send_local_descriptor));

    auto recv_data = recv_value_descriptor->value();

    EXPECT_EQ(send_data, recv_data);
}

TEST_F(TestNetwork, TransferFullDescriptors)
{
    static_assert(codable::is_static_decodable_v<TransferObject>);

    std::stop_source stop_source;

    auto progress_fiber = boost::fibers::fiber(
        [this](std::stop_token stop_token) {
            while (!stop_token.stop_requested())
            {
                if (!m_resources->progress())
                {
                    boost::this_fiber::yield();
                }
            }
        },
        stop_source.get_token());

    auto block_provider = std::make_shared<memory::memory_block_provider>();

    TransferObject send_data = {"test", 42, std::vector<int>(1_KiB)};

    auto send_data_copy = send_data;

    // Create a value descriptor
    auto value_descriptor = runtime::TypedValueDescriptor<decltype(send_data)>::create(std::move(send_data));

    // Convert to a local descriptor
    auto send_local_descriptor = runtime::LocalDescriptor2::from_value(std::move(value_descriptor), block_provider);

    // Convert the local memory blocks into remote memory blocks
    auto send_remote_descriptor = runtime::RemoteDescriptor2::from_local(std::move(send_local_descriptor),
                                                                         *m_resources);

    send_remote_descriptor->encoded_object().set_source_address(
        PortAddress2(static_cast<uint16_t>(m_resources->get_instance_id()), 0, 0, 0).combined);
    send_remote_descriptor->encoded_object().set_destination_address(
        PortAddress2(static_cast<uint16_t>(m_resources->get_instance_id()), 0, 0, 0).combined);

    auto send_remote_descriptor_object_id = send_remote_descriptor->encoded_object().object_id();

    // Check that remote payloads were registered with `DataPlaneResources2` with the correct number of tokens.
    EXPECT_EQ(send_remote_descriptor->encoded_object().tokens(), std::numeric_limits<uint64_t>::max());

    // Get the serialized data
    auto serialized_data   = send_remote_descriptor->to_bytes(memory::malloc_memory_resource::instance());
    send_remote_descriptor = nullptr;

    auto send_request = m_resources->am_send_async(m_loopback_endpoint,
                                                   serialized_data,
                                                   ucxx::AmReceiverCallbackInfo("MRC", 1 << 2));

    // while (!send_request->isCompleted())
    // {
    //     boost::this_fiber::yield();
    // }

    // while (m_resources->progress())
    // {
    //     // Do nothing
    // }

    // auto receive_request = m_resources->am_recv_async(m_loopback_endpoint);

    // Acquire the registered descriptor as a `weak_ptr` which we can use to immediately verify to be valid, but
    // invalid once `DataPlaneResources2` releases it.
    std::weak_ptr<runtime::RemoteDescriptorImpl2> registered_send_remote_descriptor = m_resources->get_descriptor(
        send_remote_descriptor_object_id);
    EXPECT_NE(registered_send_remote_descriptor.lock(), nullptr);

    // Create a remote descriptor from the received data
    // auto recv_remote_descriptor = runtime::RemoteDescriptor2::from_bytes({receive_request->getRecvBuffer()->data(),
    //                                                                       receive_request->getRecvBuffer()->getSize(),
    //                                                                       mrc::memory::memory_kind::host});

    std::unique_ptr<runtime::RemoteDescriptor2> recv_remote_descriptor;

    // Get the sent descriptor
    EXPECT_EQ(m_resources->get_inbound_channel().await_read(recv_remote_descriptor), channel::Status::success);

    auto recv_remote_descriptor_object_id = recv_remote_descriptor->encoded_object().object_id();

    // Convert to a local descriptor
    auto recv_local_descriptor = runtime::LocalDescriptor2::from_remote(std::move(recv_remote_descriptor),
                                                                        *m_resources);

    EXPECT_EQ(send_remote_descriptor_object_id, recv_remote_descriptor_object_id);

    // TODO(Peter): This is now completely async and we must progress the worker, we need a timeout in case it fails to
    // complete.
    // Wait for remote decrement messages.
    while (registered_send_remote_descriptor.lock() != nullptr)
        boost::this_fiber::yield();

    // Redundant with the above, but clarify intent.
    EXPECT_EQ(registered_send_remote_descriptor.lock(), nullptr);

    // Convert back into the value descriptor
    auto recv_value_descriptor = runtime::TypedValueDescriptor<decltype(send_data)>::from_local(
        std::move(recv_local_descriptor));

    // Finally, get the value
    const auto& recv_data = recv_value_descriptor->value();

    EXPECT_EQ(send_data_copy, recv_data);

    // Shutdown
    stop_source.request_stop();

    progress_fiber.join();
}

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

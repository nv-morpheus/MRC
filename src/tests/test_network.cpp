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

#include "internal/data_plane/client.hpp"
#include "internal/data_plane/resources.hpp"
#include "internal/memory/device_resources.hpp"
#include "internal/memory/host_resources.hpp"
#include "internal/network/resources.hpp"
#include "internal/resources/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/system/system.hpp"
#include "internal/system/system_provider.hpp"
#include "internal/ucx/memory_block.hpp"
#include "internal/ucx/registration_cache.hpp"

#include "srf/core/bitmap.hpp"
#include "srf/memory/adaptors.hpp"
#include "srf/memory/buffer.hpp"
#include "srf/memory/literals.hpp"
#include "srf/memory/resources/arena_resource.hpp"
#include "srf/memory/resources/host/pinned_memory_resource.hpp"
#include "srf/memory/resources/logging_resource.hpp"
#include "srf/memory/resources/memory_resource.hpp"
#include "srf/options/options.hpp"
#include "srf/options/placement.hpp"
#include "srf/options/resources.hpp"

#include <gtest/gtest.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <thread>
#include <utility>
#include <vector>

using namespace srf;
using namespace srf::memory::literals;

static std::shared_ptr<internal::system::System> make_system(std::function<void(Options&)> updater = nullptr)
{
    auto options = std::make_shared<Options>();
    if (updater)
    {
        updater(*options);
    }

    return internal::system::make_system(std::move(options));
}

class TestNetwork : public ::testing::Test
{};

TEST_F(TestNetwork, Arena)
{
    std::shared_ptr<srf::memory::memory_resource> mr;
    auto pinned  = std::make_shared<srf::memory::pinned_memory_resource>();
    mr           = pinned;
    auto logging = srf::memory::make_shared_resource<srf::memory::logging_resource>(mr, "pinned", 10);
    auto arena   = srf::memory::make_shared_resource<srf::memory::arena_resource>(logging, 128_MiB, 512_MiB);
    auto f       = srf::memory::make_shared_resource<srf::memory::logging_resource>(arena, "arena", 10);

    auto* ptr = f->allocate(1024);
    f->deallocate(ptr, 1024);
}

TEST_F(TestNetwork, ResourceManager)
{
    // using options.placement().resources_strategy(PlacementResources::Shared)
    // will test if cudaSetDevice is being properly called by the network services
    // since all network services for potentially multiple devices are colocated on a single thread
    auto resources = std::make_unique<internal::resources::Manager>(
        internal::system::SystemProvider(make_system([](Options& options) {
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

    // this is generally true, but perhaps we should not count on it
    EXPECT_LE(h_ucx_block->remote_handle_size(), d_ucx_block->remote_handle_size());

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
    auto resources = std::make_unique<internal::resources::Manager>(
        internal::system::SystemProvider(make_system([](Options& options) {
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
    r0.client().register_instance(1, r1.ucx_address());  // register r1 as instance_id 1
    r1.client().register_instance(0, r0.ucx_address());  // register r0 as instance_id 0

    int src = 42;
    int dst = -1;

    internal::data_plane::Request send_req;
    internal::data_plane::Request recv_req;

    r1.client().async_recv(&dst, sizeof(int), 0, recv_req);
    r0.client().async_send(&src, sizeof(int), 0, 1, send_req);

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
    auto resources = std::make_unique<internal::resources::Manager>(
        internal::system::SystemProvider(make_system([](Options& options) {
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
    r0.client().register_instance(1, r1.ucx_address());  // register r1 as instance_id 1
    r1.client().register_instance(0, r0.ucx_address());  // register r0 as instance_id 0

    internal::data_plane::Request get_req;

    r1.client().async_get(dst.data(), 1_MiB, 0, src.data(), src_keys, get_req);

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

// TEST_F(TestNetwork, NetworkEventsManagerLifeCycle)
// {
//     auto launcher = m_launch_control->prepare_launcher(std::move(m_mutable_nem));

//     // auto& service = m_launch_control->service(runnable::SrfService::data_plane::Server);
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

//     node::make_edge(m_mutable_nem->deserialize_source().source(0), *sink_0);
//     node::make_edge(m_mutable_nem->deserialize_source().source(1), *sink_1);

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

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

#include "internal/memory/callback_adaptor.hpp"
#include "internal/memory/transient_pool.hpp"
#include "internal/ucx/context.hpp"
#include "internal/ucx/memory_block.hpp"
#include "internal/ucx/registration_cache.hpp"
#include "internal/ucx/registration_resource.hpp"

#include "srf/memory/adaptors.hpp"
#include "srf/memory/buffer.hpp"
#include "srf/memory/literals.hpp"
#include "srf/memory/memory_kind.hpp"
#include "srf/memory/resources/arena_resource.hpp"
#include "srf/memory/resources/device/cuda_malloc_resource.hpp"
#include "srf/memory/resources/host/malloc_memory_resource.hpp"
#include "srf/memory/resources/host/pinned_memory_resource.hpp"
#include "srf/memory/resources/logging_resource.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <type_traits>
#include <utility>
// iwyu thinks spdlog, map, set, thread & vector are needed for arena_resource
// IWYU pragma: no_include <spdlog/sinks/basic_file_sink.h>
// IWYU pragma: no_include "spdlog/sinks/basic_file_sink.h"
// IWYU pragma: no_include <map>
// IWYU pragma: no_include <set>
// IWYU pragma: no_include <thread>
// IWYU pragma: no_include <vector>

using namespace srf;
using namespace memory;
using namespace memory::literals;

class TestMemory : public ::testing::Test
{
  protected:
};

TEST_F(TestMemory, UcxRegisterePinnedMemoryArena)
{
    auto context  = std::make_shared<internal::ucx::Context>();
    auto regcache = std::make_shared<internal::ucx::RegistrationCache>(context);

    auto pinned    = std::make_unique<pinned_memory_resource>();
    auto logger    = memory::make_unique_resource<logging_resource>(std::move(pinned), "pinned_resource");
    auto ucx       = memory::make_shared_resource<internal::ucx::RegistrationResource>(std::move(logger), regcache, 0);
    auto arena     = memory::make_shared_resource<arena_resource>(ucx, 64_MiB);
    auto arena_log = memory::make_shared_resource<logging_resource>(arena, "arena_resource");

    auto md = buffer(1_MiB, arena_log);

    auto ucx_block = ucx->registration_cache().lookup(md.data());

    EXPECT_TRUE(ucx_block);
    EXPECT_TRUE(ucx_block->local_handle());
    EXPECT_TRUE(ucx_block->remote_handle());
    EXPECT_TRUE(ucx_block->remote_handle_size());

    VLOG(1) << "ucx rbuffer size: " << ucx_block->remote_handle_size();
}

TEST_F(TestMemory, UcxRegisteredCudaMemoryArena)
{
    auto context  = std::make_shared<internal::ucx::Context>();
    auto regcache = std::make_shared<internal::ucx::RegistrationCache>(context);

    auto cuda      = std::make_unique<cuda_malloc_resource>(0);
    auto logger    = memory::make_unique_resource<logging_resource>(std::move(cuda), "cuda_resource");
    auto ucx       = memory::make_shared_resource<internal::ucx::RegistrationResource>(std::move(logger), regcache, 0);
    auto arena     = memory::make_shared_resource<arena_resource>(ucx, 64_MiB);
    auto arena_log = memory::make_shared_resource<logging_resource>(arena, "arena_resource");

    auto md = buffer(1_MiB, arena_log);

    auto ucx_block = ucx->registration_cache().lookup(md.data());

    EXPECT_TRUE(ucx_block);
    EXPECT_TRUE(ucx_block->local_handle());
    EXPECT_TRUE(ucx_block->remote_handle());
    EXPECT_TRUE(ucx_block->remote_handle_size());

    VLOG(1) << "ucx rbuffer size: " << ucx_block->remote_handle_size();
}

TEST_F(TestMemory, CallbackAdaptor)
{
    internal::memory::CallbackBuilder builder;

    std::atomic_size_t calls = 0;
    std::atomic_size_t bytes = 0;

    builder.register_callbacks([&calls](void* ptr, std::size_t _bytes) { calls++; },
                               [](void* ptr, std::size_t bytes) {});
    builder.register_callbacks([&bytes](void* ptr, std::size_t _bytes) { bytes += _bytes; },
                               [&bytes](void* ptr, std::size_t _bytes) { bytes -= bytes; });

    auto malloc = std::make_unique<srf::memory::malloc_memory_resource>();
    auto logger = srf::memory::make_unique_resource<srf::memory::logging_resource>(std::move(malloc), "malloc");
    auto callback =
        srf::memory::make_shared_resource<internal::memory::CallbackAdaptor>(std::move(logger), std::move(builder));

    EXPECT_EQ(calls, 0);
    EXPECT_EQ(bytes, 0);

    auto buffer = srf::memory::buffer(1_MiB, callback);

    EXPECT_EQ(calls, 1);
    EXPECT_EQ(bytes, 1_MiB);

    buffer.release();

    EXPECT_EQ(calls, 1);
    EXPECT_EQ(bytes, 0);
}

struct StaticData
{
    std::array<std::byte, 4_MiB> array;
};

class TickOnDestruct
{
  public:
    TickOnDestruct(int& int_ref) : m_int_ref(int_ref) {}
    ~TickOnDestruct()
    {
        m_int_ref = 42;
    }

    const int& val() const
    {
        return m_int_ref;
    }

  private:
    int& m_int_ref;
};

TEST_F(TestMemory, TransientPool)
{
    internal::memory::CallbackBuilder builder;

    std::atomic_size_t calls = 0;
    std::atomic_size_t bytes = 0;

    builder.register_callbacks([&calls](void* ptr, std::size_t _bytes) { calls++; },
                               [](void* ptr, std::size_t bytes) {});
    builder.register_callbacks([&bytes](void* ptr, std::size_t _bytes) { bytes += _bytes; },
                               [&bytes](void* ptr, std::size_t _bytes) { bytes -= bytes; });

    auto malloc = std::make_unique<srf::memory::malloc_memory_resource>();
    auto logger = srf::memory::make_unique_resource<srf::memory::logging_resource>(std::move(malloc), "malloc");
    auto callback =
        srf::memory::make_shared_resource<internal::memory::CallbackAdaptor>(std::move(logger), std::move(builder));

    internal::memory::TransientPool pool(10_MiB, 4, 8, callback);

    EXPECT_ANY_THROW(pool.await_buffer(11_MiB));

    // this should get the starting address of each block
    std::vector<void*> starting_addr;
    for (int i = 0; i < 4; i++)
    {
        auto buffer = pool.await_buffer(6_MiB);
        starting_addr.push_back(buffer.data());
    }

    // this should get the starting address of each block
    // the second pass should have the starting addresses
    std::vector<void*> addrs;
    for (int i = 0; i < 4; i++)
    {
        auto buffer = pool.await_buffer(6_MiB);
        addrs.push_back(buffer.data());
        EXPECT_TRUE(addrs.at(i) == starting_addr.at(i));
    }

    addrs.clear();
    for (int i = 0; i < 8; i++)
    {
        auto data = pool.await_object<StaticData>();
        addrs.push_back(data->array.data());

        if (i % 2 == 0)
        {
            EXPECT_TRUE(addrs.at(i) == starting_addr.at(i / 2));
        }
    }

    auto buffer = pool.await_buffer(6_MiB);

    // default constructible
    internal::memory::TransientBuffer other;

    // move and test void* gets properly nullified
    other = std::move(buffer);
    EXPECT_EQ(buffer.data(), nullptr);
    other.release();

    // construct an object TickOnDestruct on the memory backed by a TransientBuffer
    int some_int = 4;
    auto tick    = pool.await_object<TickOnDestruct>(some_int);
    EXPECT_TRUE(tick);
    EXPECT_EQ(some_int, 4);
    EXPECT_EQ(tick->val(), 4);

    // valid move assignment nullifies the original tick
    // ensure the TickOnDestruct destructor should not have been called
    auto other_tick = std::move(tick);
    EXPECT_FALSE(tick);
    EXPECT_TRUE(other_tick);
    EXPECT_EQ(some_int, 4);
    EXPECT_EQ(other_tick->val(), 4);

    // now we realise the object holding a valid TickOnDestruct,
    // this time, the destructor should modify the reference it has been holding
    other_tick.release();
    EXPECT_FALSE(other_tick);
    EXPECT_EQ(some_int, 42);
}

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

#include "test_srf.hpp"  // IWYU pragma: associated

#include "srf/cuda/common.hpp"  // for SRF_CHECK_CUDA
#include "srf/memory/adaptors.hpp"
#include "srf/memory/blob.hpp"
#include "srf/memory/block.hpp"
#include "srf/memory/buffer.hpp"
#include "srf/memory/buffer_utils.hpp"
#include "srf/memory/core/ucx_memory_block.hpp"
#include "srf/memory/literals.hpp"
#include "srf/memory/memory.hpp"  // for DeviceResourceView, HostResourceView
#include "srf/memory/memory_kind.hpp"
#include "srf/memory/resource_view.hpp"
#include "srf/memory/resources/arena_resource.hpp"
#include "srf/memory/resources/device/cuda_malloc_resource.hpp"
#include "srf/memory/resources/host/malloc_memory_resource.hpp"
#include "srf/memory/resources/host/pinned_memory_resource.hpp"
#include "srf/memory/resources/logging_resource.hpp"
// #include "srf/memory/resources/ucx_registered_resource.hpp"
#include "internal/ucx/context.hpp"

#include <cuda_runtime.h>  // for cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize, CUstream_st, cudaStream_t
#include <cuda/memory_resource>

#include <memory>
#include <ostream>      // for logging
#include <type_traits>  // for remove_reference<>::type implied by blob mblob(std::move(b));
#include <utility>      // for move
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

TEST_CLASS(Memory);

TEST_F(TestMemory, ucx_registration_resource)
{
    auto context   = std::make_shared<ucx::Context>();
    auto pinned    = std::make_unique<pinned_memory_resource>();
    auto logger    = memory::make_unique_resource<logging_resource>(std::move(pinned), "pinned_resource");
    auto ucx       = memory::make_shared_resource<ucx_registered_resource>(std::move(logger), context);
    auto arena     = memory::make_shared_resource<arena_resource>(ucx, 64_MiB);
    auto arena_log = memory::make_shared_resource<logging_resource>(arena, "arena_resource");

    using host_buffer = buffer<cuda::memory_access::host, cuda::memory_access::device>;

    auto md = host_buffer(1_MiB, arena_log);

    auto ucx_block = ucx->lookup(md.data());

    CHECK(ucx_block.local_handle());
    CHECK(ucx_block.remote_handle());
    CHECK(ucx_block.remote_handle_size());

    LOG(INFO) << "ucx rbuffer size: " << ucx_block.remote_handle_size();
}

TEST_F(TestMemory, ucx_registration_resource_cuda)
{
    auto context   = std::make_shared<ucx::Context>();
    auto cuda      = std::make_unique<cuda_malloc_resource>(0);
    auto logger    = memory::make_unique_resource<logging_resource>(std::move(cuda), "cuda_resource");
    auto ucx       = memory::make_shared_resource<ucx_registered_resource>(std::move(logger), context);
    auto arena     = memory::make_shared_resource<arena_resource>(ucx, 64_MiB);
    auto arena_log = memory::make_shared_resource<logging_resource>(arena, "arena_resource");

    auto pinned_logger =
        memory::make_shared_resource<logging_resource>(std::make_unique<pinned_memory_resource>(), "pinned_resource");

    using buffer_type = buffer<cuda::memory_access::device>;

    auto md = buffer_type(1_MiB, arena_log);

    auto ucx_block = ucx->lookup(md.data());

    CHECK(ucx_block.local_handle());
    CHECK(ucx_block.remote_handle());
    CHECK(ucx_block.remote_handle_size());

    LOG(INFO) << "ucx rbuffer size: " << ucx_block.remote_handle_size();

    auto pb = buffer_type(2_MiB, pinned_logger);
}

TEST_F(TestMemory, resource_view_with_raw_pointer)
{
    pinned_memory_resource pinned;
    resource_view<cuda::memory_access::host> view(&pinned);

    EXPECT_EQ(view.kind(), memory_kind_type::pinned);
    EXPECT_EQ(view.prv()->kind(), memory_kind_type::pinned);
}

TEST_F(TestMemory, resource_view_with_shared_pointer)
{
    auto pinned = std::make_shared<pinned_memory_resource>();
    resource_view<cuda::memory_access::host> view(pinned);

    EXPECT_EQ(view.kind(), memory_kind_type::pinned);
    EXPECT_EQ(view.prv()->kind(), memory_kind_type::pinned);
}

TEST_F(TestMemory, buffer_with_raw_pointer)
{
    pinned_memory_resource pinned;

    buffer<cuda::memory_access::host> b(1_MiB, &pinned);
    const_block view_from_buffer(b);
    block mv(b);

    blob mblob(std::move(b));
    const_block view_from_blob(mblob);

    auto oblob = mblob.allocate(2_MiB);
}

TEST_F(TestMemory, buffer_with_smart_pointer)
{
    auto pinned = std::make_shared<pinned_memory_resource>();
    buffer<cuda::memory_access::host> b(1_MiB, pinned);

    const_block blob(b);
}

TEST_F(TestMemory, OldAPI)
{
    auto context   = std::make_shared<ucx::Context>();
    auto pinned    = std::make_unique<pinned_memory_resource>();
    auto logger    = memory::make_unique_resource<logging_resource>(std::move(pinned), "pinned_resource");
    auto ucx       = memory::make_shared_resource<ucx_registered_resource>(std::move(logger), context);
    auto arena     = memory::make_shared_resource<arena_resource>(ucx, 64_MiB);
    auto arena_log = memory::make_shared_resource<logging_resource>(arena, "arena_resource");

    OldHostAllocator host(arena_log, ucx);

    auto ialloc = host.shared();

    auto md = ialloc->allocate_descriptor(2_MiB);

    EXPECT_EQ(md.type(), memory_kind_type::pinned);

    blob blob(std::move(md));
}

TEST_F(TestMemory, BlobEmptyBoolOperator)
{
    blob b;

    EXPECT_TRUE(b.empty());
    EXPECT_FALSE(b);
}

TEST_F(TestMemory, Copy)
{
    auto malloc = std::make_shared<memory::malloc_memory_resource>();
    auto pinned = std::make_shared<memory::pinned_memory_resource>();
    auto device = std::make_shared<memory::cuda_malloc_resource>(0);

    auto mb = buffer<::cuda::memory_access::host>(1_MiB, malloc);
    auto pb = buffer(2_MiB, HostResourceView(pinned));
    auto db = buffer(4_MiB, DeviceResourceView(device));

    buffer_utils::copy(mb, pb, 1_MiB);
    buffer_utils::copy(pb, mb, 1_MiB);

    EXPECT_DEATH(buffer_utils::copy(mb, pb, 2_MiB), "");

    // these should not compile
    // buffer_utils::copy(mb, db, 1_MiB);
    // buffer_utils::copy(db, pb, 1_MiB);
}

TEST_F(TestMemory, AsyncCopy)
{
    auto malloc = std::make_shared<memory::malloc_memory_resource>();
    auto pinned = std::make_shared<memory::pinned_memory_resource>();
    auto device = std::make_shared<memory::cuda_malloc_resource>(0);

    auto mb = buffer<::cuda::memory_access::host>(1_MiB, malloc);
    auto pb = buffer(2_MiB, HostResourceView(pinned));
    auto db = buffer(4_MiB, DeviceResourceView(device));

    cudaStream_t stream;
    SRF_CHECK_CUDA(cudaStreamCreate(&stream));

    // should not compile
    // buffer_utils::async_copy(mb, pb, 1_MiB, stream);
    // buffer_utils::async_copy(db, mb, 1_MiB, stream);

    // these should not compile
    buffer_utils::async_copy(pb, db, 1_MiB, stream);
    buffer_utils::async_copy(db, pb, 1_MiB, stream);

    SRF_CHECK_CUDA(cudaStreamSynchronize(stream));
    SRF_CHECK_CUDA(cudaStreamDestroy(stream));
}

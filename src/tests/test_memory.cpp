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

#include "internal/ucx/context.hpp"
#include "internal/ucx/memory_block.hpp"
#include "internal/ucx/registration_cache.hpp"
#include "internal/ucx/registration_resource.hpp"

#include "srf/cuda/common.hpp"
#include "srf/memory/adaptors.hpp"
#include "srf/memory/buffer.hpp"
#include "srf/memory/buffer_view.hpp"
#include "srf/memory/literals.hpp"
#include "srf/memory/memory_kind.hpp"
#include "srf/memory/resources/arena_resource.hpp"
#include "srf/memory/resources/device/cuda_malloc_resource.hpp"
#include "srf/memory/resources/host/malloc_memory_resource.hpp"
#include "srf/memory/resources/host/pinned_memory_resource.hpp"
#include "srf/memory/resources/logging_resource.hpp"

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <memory>
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

TEST_F(TestMemory, ucx_registration_resource)
{
    auto context  = std::make_shared<internal::ucx::Context>();
    auto regcache = std::make_shared<internal::ucx::RegistrationCache>(context);

    auto pinned    = std::make_unique<pinned_memory_resource>();
    auto logger    = memory::make_unique_resource<logging_resource>(std::move(pinned), "pinned_resource");
    auto ucx       = memory::make_shared_resource<internal::ucx::RegistrationResource>(std::move(logger), regcache);
    auto arena     = memory::make_shared_resource<arena_resource>(ucx, 64_MiB);
    auto arena_log = memory::make_shared_resource<logging_resource>(arena, "arena_resource");

    auto md = buffer(1_MiB, arena_log);

    auto ucx_block = ucx->registration_cache().lookup(md.data());

    CHECK(ucx_block.local_handle());
    CHECK(ucx_block.remote_handle());
    CHECK(ucx_block.remote_handle_size());

    VLOG(1) << "ucx rbuffer size: " << ucx_block.remote_handle_size();
}

TEST_F(TestMemory, ucx_registration_resource_cuda)
{
    auto context  = std::make_shared<internal::ucx::Context>();
    auto regcache = std::make_shared<internal::ucx::RegistrationCache>(context);

    auto cuda      = std::make_unique<cuda_malloc_resource>(0);
    auto logger    = memory::make_unique_resource<logging_resource>(std::move(cuda), "cuda_resource");
    auto ucx       = memory::make_shared_resource<internal::ucx::RegistrationResource>(std::move(logger), regcache);
    auto arena     = memory::make_shared_resource<arena_resource>(ucx, 64_MiB);
    auto arena_log = memory::make_shared_resource<logging_resource>(arena, "arena_resource");

    auto md = buffer(1_MiB, arena_log);

    auto ucx_block = ucx->registration_cache().lookup(md.data());

    CHECK(ucx_block.local_handle());
    CHECK(ucx_block.remote_handle());
    CHECK(ucx_block.remote_handle_size());

    LOG(INFO) << "ucx rbuffer size: " << ucx_block.remote_handle_size();
}

// TEST_F(TestMemory, Copy)
// {
//     auto malloc = std::make_shared<memory::malloc_memory_resource>();
//     auto pinned = std::make_shared<memory::pinned_memory_resource>();
//     auto device = std::make_shared<memory::cuda_malloc_resource>(0);

//     auto mb = buffer<::cuda::memory_access::host>(1_MiB, malloc);
//     auto pb = buffer(2_MiB, HostResourceView(pinned));
//     auto db = buffer(4_MiB, DeviceResourceView(device));

//     buffer_utils::copy(mb, pb, 1_MiB);
//     buffer_utils::copy(pb, mb, 1_MiB);

//     EXPECT_DEATH(buffer_utils::copy(mb, pb, 2_MiB), "");

//     // these should not compile
//     // buffer_utils::copy(mb, db, 1_MiB);
//     // buffer_utils::copy(db, pb, 1_MiB);
// }

// TEST_F(TestMemory, AsyncCopy)
// {
//     auto malloc = std::make_shared<memory::malloc_memory_resource>();
//     auto pinned = std::make_shared<memory::pinned_memory_resource>();
//     auto device = std::make_shared<memory::cuda_malloc_resource>(0);

//     auto mb = buffer<::cuda::memory_access::host>(1_MiB, malloc);
//     auto pb = buffer(2_MiB, HostResourceView(pinned));
//     auto db = buffer(4_MiB, DeviceResourceView(device));

//     cudaStream_t stream;
//     SRF_CHECK_CUDA(cudaStreamCreate(&stream));

//     // should not compile
//     // buffer_utils::async_copy(mb, pb, 1_MiB, stream);
//     // buffer_utils::async_copy(db, mb, 1_MiB, stream);

//     // these should not compile
//     buffer_utils::async_copy(pb, db, 1_MiB, stream);
//     buffer_utils::async_copy(db, pb, 1_MiB, stream);

//     SRF_CHECK_CUDA(cudaStreamSynchronize(stream));
//     SRF_CHECK_CUDA(cudaStreamDestroy(stream));
// }

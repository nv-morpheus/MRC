/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <srf/cuda/common.hpp>
#include <srf/cuda/device_guard.hpp>
#include <srf/memory/resources/memory_resource.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace srf::memory {

class cuda_malloc_resource final : public memory_resource<::cuda::memory_kind::device>
{
  public:
    cuda_malloc_resource(int device_id) : memory_resource("cuda_malloc"), m_device_id(device_id) {}
    ~cuda_malloc_resource() override = default;

  private:
    void* do_allocate(std::size_t bytes, std::size_t /*__alignment*/) final
    {
        void* ptr = nullptr;
        DeviceGuard guard(m_device_id);
        SRF_CHECK_CUDA(cudaMalloc(&ptr, bytes));
        return ptr;
    }

    void do_deallocate(void* ptr, std::size_t /*__bytes*/, std::size_t /*__alignment*/) final
    {
        DeviceGuard guard(m_device_id);
        SRF_CHECK_CUDA(cudaFree(ptr));
    }

    memory_kind_type do_kind() const final
    {
        return memory_kind_type::device;
    }

    int m_device_id;
};

}  // namespace srf::memory

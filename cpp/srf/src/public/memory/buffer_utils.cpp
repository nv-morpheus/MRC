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

#include <srf/cuda/common.hpp>
#include <srf/memory/buffer_utils.hpp>
#include <srf/memory/memory_kind.hpp>

namespace srf::memory {

void buffer_utils::copy(blob& dst, const blob& src, std::size_t bytes)
{
    DCHECK_LE(bytes, dst.bytes());
    DCHECK_LE(bytes, src.bytes());

    if (src.kind() == memory_kind_type::host || dst.kind() == memory_kind_type::host)
    {
        if (src.kind() == dst.kind())
        {
            std::memcpy(dst.data(), src.data(), bytes);
            return;
        }

        throw std::runtime_error("buffered host_memory copies are not implemented");
    }

    throw std::runtime_error("synchronous cuda memory copies are not implemented");
}

void buffer_utils::async_copy(blob& dst, const blob& src, std::size_t bytes, cudaStream_t stream)
{
    DCHECK_LE(bytes, dst.bytes());
    DCHECK_LE(bytes, src.bytes());

    if (src.kind() == memory_kind_type::host || dst.kind() == memory_kind_type::host)
    {
        throw std::runtime_error("asynchronous host memory copies are not implemented");
    }

    SRF_CHECK_CUDA(cudaMemcpyAsync(dst.data(), src.data(), bytes, cudaMemcpyDefault));
}

}  // namespace srf::memory

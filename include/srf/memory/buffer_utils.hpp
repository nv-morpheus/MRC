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

#pragma once

#include "srf/cuda/common.hpp"
#include "srf/memory/blob.hpp"

namespace srf::memory {

struct buffer_utils  // NOLINT
{
    /**
     * @brief Copy bytes from one blob to another.
     *
     * @param dst
     * @param src
     * @param bytes
     */
    static void copy(blob& dst, const blob& src, std::size_t bytes);

    /**
     * @brief Asynchronously copy bytes from one blob to another on the specified CUDA stream
     *
     * @param dst
     * @param src
     * @param bytes
     * @param stream
     */
    static void async_copy(blob& dst, const blob& src, std::size_t bytes, cudaStream_t stream);

    /**
     * @brief Copies data from the source to the destination buffer, where both buffers have the property
     * cuda::memory_access::host
     *
     * @tparam DstProperties
     * @tparam SrcProperties
     * @param dst
     * @param src
     * @param bytes
     */
    template <typename... DstProperties, typename... SrcProperties>
    static void copy(buffer<DstProperties...>& dst, const buffer<SrcProperties...>& src, std::size_t bytes)
    {
        static_assert(buffer<DstProperties...>::template contains<::cuda::memory_access::host>::value &&
                          buffer<SrcProperties...>::template contains<::cuda::memory_access::host>::value,
                      "host access only");
        CHECK_LE(bytes, dst.bytes());
        CHECK_LE(bytes, src.bytes());
        std::memcpy(dst.data(), src.data(), bytes);
    }

    /**
     * @brief Copies data asynchronously from the source to the destination buffer on the provided stream. Both buffers
     * are required to have the cuda::memory_access:device property.
     *
     * @tparam DstProperties
     * @tparam SrcProperties
     * @param dst
     * @param src
     * @param bytes
     * @param stream
     */
    template <typename... DstProperties, typename... SrcProperties>
    static void async_copy(buffer<DstProperties...>& dst,
                           const buffer<SrcProperties...>& src,
                           std::size_t bytes,
                           cudaStream_t stream)
    {
        static_assert(buffer<DstProperties...>::template contains<::cuda::memory_access::device>::value &&
                          buffer<SrcProperties...>::template contains<::cuda::memory_access::device>::value,
                      "device access only");
        CHECK_LE(bytes, dst.bytes());
        CHECK_LE(bytes, src.bytes());
        SRF_CHECK_CUDA(cudaMemcpyAsync(dst.data(), src.data(), bytes, cudaMemcpyDefault, stream));
    }
};

}  // namespace srf::memory

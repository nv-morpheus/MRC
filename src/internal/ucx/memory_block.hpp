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

#include "internal/memory/memory_block.hpp"
#include "internal/ucx/context.hpp"

#include <glog/logging.h>

namespace srf::internal::ucx {

/**
 * @brief Extends memory::MemoryBlock to add UCX memory registration information
 */
struct MemoryBlock : public memory::MemoryBlock
{
  public:
    MemoryBlock() = default;
    MemoryBlock(void* data, std::size_t bytes);
    MemoryBlock(
        void* data, std::size_t bytes, ucp_mem_h local_handle, void* remote_handle, std::size_t remote_handle_size);
    MemoryBlock(const MemoryBlock& block, ucp_mem_h local_handle, void* remote_handle, std::size_t remote_handle_size);
    ~MemoryBlock() override = default;

    /**
     * @brief UCX local memory handle / access key
     */
    ucp_mem_h local_handle() const;

    /**
     * @brief Starting address to the UCX remote memory handle / access key
     *
     * The remote handle / remote keys can vary in length based on the transports in use. Use `remote_handle_size` to
     * query the size of the contiguous buffer started with the returned address.
     */
    void* remote_handle() const;

    /**
     * @brief Size in bytes of the contiguous memory buffer pointed to by `remote_handle()`
     */
    std::size_t remote_handle_size() const;

    std::string packed_remote_keys() const;

  private:
    ucp_mem_h m_local_handle{nullptr};
    void* m_remote_handle{nullptr};
    std::size_t m_remote_handle_size{0};
};

}  // namespace srf::internal::ucx

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

#include "internal/memory/block_manager.hpp"
#include "internal/ucx/context.hpp"
#include "internal/ucx/memory_block.hpp"

#include <glog/logging.h>

#include <memory>
#include <mutex>
#include <optional>

namespace srf::internal::ucx {

/**
 * @brief UCX Registration Cache
 *
 * UCX memory registration object that will both register/deregister memory as well as cache the set of local and remote
 * keys for each registration. The cache can be queried for the original memory block by providing any valid address
 * contained in the contiguous block.
 */
class RegistrationCache final
{
  public:
    RegistrationCache(std::shared_ptr<ucx::Context> context) : m_context(std::move(context))
    {
        CHECK(m_context);
    }

    /**
     * @brief Register a contiguous block of memory starting at addr and spanning `bytes` bytes.
     *
     * For each block of memory registered with the RegistrationCache, an entry containing the block information is
     * storage and can be queried.
     *
     * @param addr
     * @param bytes
     */
    void add_block(void* addr, std::size_t bytes)
    {
        DCHECK(addr && bytes);
        auto [lkey, rkey, rkey_size] = m_context->register_memory_with_rkey(addr, bytes);
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        m_blocks.add_block({addr, bytes, lkey, rkey, rkey_size});
    }

    /**
     * @brief Deregister a contiguous block of memory from the ucx context and remove the cache entry
     *
     * @param addr
     * @param bytes
     * @return std::size_t
     */
    std::size_t drop_block(const void* addr, std::size_t bytes)
    {
        const auto* block = m_blocks.find_block(addr);
        CHECK(block);
        bytes = block->bytes();
        m_context->unregister_memory(block->local_handle(), block->remote_handle());
        m_blocks.drop_block(addr);
        return bytes;
    }

    /**
     * @brief Look up the memory registration details for a given address.
     *
     * This method queries the registration cache to find the UcxMemoryBlock containing the original address and size as
     * well as the local and remote keys associated with the memory block.
     *
     * Any address contained within a registered block can be used to query the UcxMemoryBlock
     *
     * @param addr
     * @return const MemoryBlock&
     */
    std::optional<ucx::MemoryBlock> lookup(const void* addr) const noexcept
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        const auto* ptr = m_blocks.find_block(addr);
        if (ptr == nullptr)
        {
            return std::nullopt;
        }
        return {*ptr};
    }

  private:
    mutable std::mutex m_mutex;
    const std::shared_ptr<ucx::Context> m_context;
    memory::BlockManager<MemoryBlock> m_blocks;
};

}  // namespace srf::internal::ucx

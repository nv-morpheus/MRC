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
#include "internal/ucx/endpoint.hpp"
#include "internal/ucx/memory_block.hpp"

#include <glog/logging.h>
#include <ucp/api/ucp_def.h>

#include <memory>
#include <mutex>
#include <optional>

namespace mrc::internal::ucx {

/**
 * @brief UCX Registration Cache
 *
 * UCX memory registration object that will both register/deregister memory as well as cache the set of local and remote
 * keys for each registration. The cache can be queried for the original memory block by providing any valid address
 * contained in the contiguous block.
 */
class RemoteRegistrationCache final
{
    class MemoryBlock : public memory::MemoryBlock
    {
      public:
        MemoryBlock() = default;
        MemoryBlock(const void* data, std::size_t bytes, ucp_rkey_h rkey_handle) :
          memory::MemoryBlock(data, bytes),
          m_rkey_handle(rkey_handle)
        {}
        ~MemoryBlock() override = default;

        /**
         * @brief UCX remote key unpacked to a given endpoint
         */
        ucp_rkey_h remote_key_handle() const
        {
            return m_rkey_handle;
        }

      private:
        ucp_rkey_h m_rkey_handle;
    };

  public:
    RemoteRegistrationCache(ucp_ep_h endpoint) : m_endpoint(std::move(endpoint))
    {
        CHECK(m_endpoint);
    }
    ~RemoteRegistrationCache()
    {
        m_blocks.for_each_block([this](const MemoryBlock& block) { ucp_rkey_destroy(block.remote_key_handle()); });
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
    MemoryBlock add_block(const void* addr, std::size_t bytes, const std::string& packed_remote_key)
    {
        DCHECK(addr && bytes);
        ucp_rkey_h rkey;
        auto rc = ucp_ep_rkey_unpack(m_endpoint, packed_remote_key.data(), &rkey);
        CHECK_EQ(rc, UCS_OK);
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        MemoryBlock block{addr, bytes, rkey};
        m_blocks.add_block(block);
        return block;
    }

    /**
     * @brief Deregister a contiguous block of memory from the ucx context and remove the cache entry
     *
     * @param addr
     * @param bytes
     * @return std::size_t
     */
    void drop_block(const void* addr)
    {
        const auto* block = m_blocks.find_block(addr);
        CHECK(block);
        ucp_rkey_destroy(block->remote_key_handle());
        m_blocks.drop_block(addr);
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
    std::optional<MemoryBlock> lookup(const void* addr) const noexcept
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        const auto* ptr = m_blocks.find_block(addr);
        if (ptr == nullptr)
        {
            return std::nullopt;
        }
        return {*ptr};
    }

    ucp_ep_h endpoint() const
    {
        return m_endpoint;
    }

  private:
    mutable std::mutex m_mutex;
    const ucp_ep_h m_endpoint;
    memory::BlockManager<MemoryBlock> m_blocks;
};

}  // namespace mrc::internal::ucx

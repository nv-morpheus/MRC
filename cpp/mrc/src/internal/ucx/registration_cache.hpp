/*
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

#include "mrc/memory/memory_kind.hpp"

#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

namespace ucxx {
class Context;
class MemoryHandle;
}

namespace mrc::ucx {

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
    void add_block(const void* addr, std::size_t bytes)
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

/**
 * @brief UCX Registration Cache
 *
 * UCX memory registration object that will both register/deregister memory as well as cache the set of local and remote
 * keys for each registration. The cache can be queried for the original memory block by providing any valid address
 * contained in the contiguous block.
 */
class RegistrationCache2 final
{
  public:
    RegistrationCache2(std::shared_ptr<ucxx::Context> context);

    /**
     * @brief Register a contiguous block of memory starting at addr and spanning `bytes` bytes.
     *
     * For each block of memory registered with the RegistrationCache, an entry containing the block information is
     * storage and can be queried.
     *
     * @param addr
     * @param bytes
     */
    const ucx::MemoryBlock& add_block(const void* addr, std::size_t bytes);

    const ucx::MemoryBlock& add_block(uintptr_t addr, std::size_t bytes);

    /**
     * @brief Deregister a contiguous block of memory from the ucx context and remove the cache entry
     *
     * @param addr
     * @param bytes
     * @return std::size_t
     */
    std::size_t drop_block(const void* addr, std::size_t bytes);

    std::size_t drop_block(uintptr_t addr, std::size_t bytes);

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
    std::optional<ucx::MemoryBlock> lookup(const void* addr) const noexcept;

    std::optional<ucx::MemoryBlock> lookup(uintptr_t addr) const noexcept;

  private:
    ucp_mem_h register_memory(const void* address, std::size_t bytes);

    std::tuple<ucp_mem_h, void*, std::size_t> register_memory_with_rkey(const void* address, std::size_t bytes);

    void unregister_memory(ucp_mem_h handle, void* rbuffer = nullptr);

    mutable std::mutex m_mutex;
    const std::shared_ptr<ucxx::Context> m_context;
    memory::BlockManager<MemoryBlock> m_blocks;
};

/**
 * @brief UCX Registration Cache
 *
 * UCX memory registration object that will both register/deregister memory. The cache can be queried for the original
 * memory block by providing the id of the descriptor object and the starting address of the contiguous block.
 */
class RegistrationCache3 final
{
  public:
    RegistrationCache3(std::shared_ptr<ucxx::Context> context);

    /**
     * @brief Register a contiguous block of memory starting at addr and spanning `bytes` bytes.
     *
     * For each block of memory registered with the RegistrationCache, an entry containing the block information is
     * storage and can be queried.
     *
     * @param obj_id ID of the descriptor object that owns the memory block being registered
     * @param addr
     * @param bytes
     * @param memory_type
     * @return std::shared_ptr<ucxx::MemoryHandle>
     */
    std::shared_ptr<ucxx::MemoryHandle> add_block(uint64_t obj_id, void* addr, std::size_t bytes, memory::memory_kind memory_type);

    std::shared_ptr<ucxx::MemoryHandle> add_block(uint64_t obj_id, uintptr_t addr, std::size_t bytes, memory::memory_kind memory_type);

    /**
     * @brief Look up the memory registration details for a given address.
     *
     * This method queries the registration cache to find the MemoryHanlde containing the original address and size as
     * well as the serialized remote keys associated with the memory block.
     *
     * @param obj_id ID of the descriptor object that owns the memory block being registered
     * @param addr
     * @return std::shared_ptr<ucxx::MemoryHandle>
     */
    std::optional<std::shared_ptr<ucxx::MemoryHandle>> lookup(uint64_t obj_id, const void* addr) const noexcept;

    std::optional<std::shared_ptr<ucxx::MemoryHandle>> lookup(uint64_t obj_id, uintptr_t addr) const noexcept;

    /**
     * @brief Deregistration of all memory blocks owned by the descriptor object with id obj_id
     *
     * This method deregisters all memory blocks owned by the descriptor object at the end of the descriptor's lifetime.
     * Required so the system does not run into memory insufficiency errors.
     *
     * @param obj_id ID of the descriptor object that owns the memory block being registered
     */
    void remove_descriptor(uint64_t obj_id);

  private:
    mutable std::mutex m_mutex;
    const std::shared_ptr<ucxx::Context> m_context;

    // <descriptor object_id : <address : MemoryHandle>>
    std::map<uint64_t, std::map<const void*, std::shared_ptr<ucxx::MemoryHandle>>> m_memory_handle_by_address;
};
}  // namespace mrc::ucx

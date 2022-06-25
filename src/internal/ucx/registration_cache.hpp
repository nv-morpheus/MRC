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

namespace srf::internal::ucx {

class RegistrationCache final
{
  public:
    RegistrationCache(std::shared_ptr<ucx::Context> context) : m_context(std::move(context))
    {
        CHECK(m_context);
    }

    void add_block(void* addr, std::size_t bytes)
    {
        DCHECK(addr && bytes);
        auto [lkey, rkey, rkey_size] = m_context->register_memory_with_rkey(addr, bytes);
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        m_blocks.add_block({addr, bytes, lkey, rkey, rkey_size});
    }

    std::size_t drop_block(void* addr, std::size_t bytes)
    {
        const auto* block = m_blocks.find_block(addr);
        CHECK(block);
        bytes = block->bytes();
        m_context->unregister_memory(block->local_handle(), block->remote_handle());
        m_blocks.drop_block(addr);
        return bytes;
    }

    const MemoryBlock& lookup(void* addr) const noexcept
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        const auto* ptr = m_blocks.find_block(addr);
        CHECK(ptr);
        return *ptr;
    }

  private:
    mutable std::mutex m_mutex;
    const std::shared_ptr<ucx::Context> m_context;
    memory::BlockManager<MemoryBlock> m_blocks;
};

}  // namespace srf::internal::ucx

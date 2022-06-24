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
#include "internal/memory/ucx_memory_block.hpp"
#include "internal/ucx/context.hpp"

#include <srf/memory/adaptors.hpp>
#include <srf/memory/memory_kind.hpp>

#include <glog/logging.h>

#include <memory>
#include <mutex>

namespace srf::internal::memory {

inline ucs_memory_type_t memory_type(const srf::memory::memory_kind& type)
{
    switch (type)
    {
    case srf::memory::memory_kind::host:
    case srf::memory::memory_kind::pinned:
        return UCS_MEMORY_TYPE_HOST;
        break;
    case srf::memory::memory_kind::device:
    case srf::memory::memory_kind::managed:
        return UCS_MEMORY_TYPE_CUDA;
        break;
    default:
        throw std::runtime_error("unhandled memory_kind");
        break;
    }
    return UCS_MEMORY_TYPE_UNKNOWN;
}

struct UcxRegistrationCache
{
    virtual ~UcxRegistrationCache()                 = default;
    virtual UcxMemoryBlock lookup(void* addr) const = 0;
};

template <typename UpstreamT>
class UcxRegisteredResource final : public srf::memory::adaptor<UpstreamT>, public UcxRegistrationCache
{
  public:
    UcxRegisteredResource(UpstreamT upstream, ucx::Context& context) :
      srf::memory::adaptor<UpstreamT>(std::move(upstream)),
      m_context(context)
    {}

    UcxMemoryBlock lookup(void* addr) const final
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        const auto* ptr = m_blocks.find_block(addr);
        if (ptr == nullptr)
        {
            throw std::runtime_error("unable to lookup metadata for address");
        }
        return *ptr;
    }

  private:
    ucx::Context& m_context;
    BlockManager<UcxMemoryBlock> m_blocks;
    mutable std::mutex m_mutex;

    [[nodiscard]] void* do_allocate(std::size_t bytes, std::size_t alignment) final
    {
        void* mem = this->resource()->allocate(bytes, alignment);

        if (mem == nullptr)
        {
            return nullptr;
        }

        auto [lkey, rkey, rkey_size] = m_context.register_memory_with_rkey(mem, bytes);
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        m_blocks.add_block({mem, bytes, lkey, rkey, rkey_size});
        return mem;
    }

    void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) final
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        const auto* block = m_blocks.find_block(ptr);
        CHECK(block);
        m_context.unregister_memory(block->local_handle(), block->remote_handle());
        m_blocks.drop_block(ptr);
        this->resource()->deallocate(ptr, bytes, alignment);
    }
};

}  // namespace srf::internal::memory

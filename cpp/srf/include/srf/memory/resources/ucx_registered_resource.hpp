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

#include <srf/memory/adaptors.hpp>
#include <srf/memory/core/block_manager.hpp>
#include <srf/memory/core/ucx_memory_block.hpp>
#include "internal/ucx/context.hpp"

#include <cuda/memory_resource>

#include <glog/logging.h>

#include <memory>
#include <mutex>

namespace srf::memory {

inline ucs_memory_type_t memory_type(const memory_kind_type& type)
{
    switch (type)
    {
    case memory_kind_type::host:
    case memory_kind_type::pinned:
        return UCS_MEMORY_TYPE_HOST;
        break;
    case memory_kind_type::device:
    case memory_kind_type::managed:
        return UCS_MEMORY_TYPE_CUDA;
        break;
    default:
        throw std::runtime_error("unhandled memory_kind_type");
        break;
    }
    return UCS_MEMORY_TYPE_UNKNOWN;
}

struct ucx_registration_cache
{
    virtual ~ucx_registration_cache()                 = default;
    virtual ucx_memory_block lookup(void* addr) const = 0;
};

// template <typename Upstream>
// class ucx_registered_resource final : public upstream_resource<Upstream>, public ucx_registration_cache
// {
//   public:
//     ucx_registered_resource(Upstream upstream, std::shared_ptr<ucx::Context> context) :
//       upstream_resource<Upstream>(std::move(upstream), "ucx_registered"),
//       m_context(std::move(context))
//     {
//         CHECK(m_context) << "ucx context cannot be null";
//     }

//     ucx_memory_block lookup(void* addr) const final
//     {
//         std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//         const auto* ptr = m_blocks.find_block(addr);
//         if (ptr == nullptr)
//         {
//             throw std::runtime_error("unable to lookup metadata for address");
//         }
//         return *ptr;
//     }

//   private:
//     std::shared_ptr<ucx::Context> m_context{nullptr};
//     block_manager<ucx_memory_block> m_blocks;
//     mutable std::mutex m_mutex;

//     [[nodiscard]] void* do_allocate(std::size_t bytes, std::size_t alignment) final
//     {
//         void* mem = this->resource()->allocate(bytes, alignment);

//         if (mem == nullptr)
//         {
//             return nullptr;
//         }

//         auto [lkey, rkey, rkey_size] = m_context->register_memory_with_rkey(mem, bytes);
//         std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//         m_blocks.add_block(ucx_memory_block(mem, bytes, lkey, rkey, rkey_size));
//         return mem;
//     }

//     void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) final
//     {
//         std::lock_guard<decltype(m_mutex)> lock(m_mutex);
//         auto block = m_blocks.find_block(ptr);
//         if (block == nullptr)
//         {
//             LOG(FATAL) << "unable to lookup block";
//         }
//         m_context->unregister_memory(block->local_handle(), block->remote_handle());
//         m_blocks.drop_block(ptr);
//         this->resource()->deallocate(ptr, bytes, alignment);
//     }
// };

}  // namespace srf::memory

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

namespace srf::internal::memory {

struct UcxMemoryBlock : public MemoryBlock
{
  public:
    UcxMemoryBlock() = default;
    UcxMemoryBlock(void* data, std::size_t bytes) : MemoryBlock(data, bytes) {}
    UcxMemoryBlock(
        void* data, std::size_t bytes, ucp_mem_h local_handle, void* remote_handle, std::size_t remote_handle_size) :
      MemoryBlock(data, bytes),
      m_local_handle(local_handle),
      m_remote_handle(remote_handle),
      m_remote_handle_size(remote_handle_size)
    {
        // if either remote handle or size are set, both have to be valid
        if ((m_remote_handle != nullptr) || (m_remote_handle_size != 0U))
        {
            CHECK(m_remote_handle && m_remote_handle_size);
        }
    }
    UcxMemoryBlock(const UcxMemoryBlock& block,
                   ucp_mem_h local_handle,
                   void* remote_handle,
                   std::size_t remote_handle_size) :
      MemoryBlock(block),
      m_local_handle(local_handle),
      m_remote_handle(remote_handle),
      m_remote_handle_size(remote_handle_size)
    {
        if ((m_remote_handle != nullptr) || (m_remote_handle_size != 0U))
        {
            CHECK(m_remote_handle && m_remote_handle_size);
        }
    }
    ~UcxMemoryBlock() override = default;

    ucp_mem_h local_handle() const
    {
        return m_local_handle;
    }
    void* remote_handle() const
    {
        return m_remote_handle;
    }
    std::size_t remote_handle_size() const
    {
        return m_remote_handle_size;
    }

  private:
    ucp_mem_h m_local_handle{nullptr};
    void* m_remote_handle{nullptr};
    std::size_t m_remote_handle_size{0};
};

}  // namespace srf::internal::memory

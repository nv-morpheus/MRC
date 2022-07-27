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

#include "internal/ucx/memory_block.hpp"

namespace srf::internal::ucx {

MemoryBlock::MemoryBlock(void* data, std::size_t bytes) : memory::MemoryBlock(data, bytes) {}

MemoryBlock::MemoryBlock(
    void* data, std::size_t bytes, ucp_mem_h local_handle, void* remote_handle, std::size_t remote_handle_size) :
  memory::MemoryBlock(data, bytes),
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

MemoryBlock::MemoryBlock(const MemoryBlock& block,
                         ucp_mem_h local_handle,
                         void* remote_handle,
                         std::size_t remote_handle_size) :
  memory::MemoryBlock(block),
  m_local_handle(local_handle),
  m_remote_handle(remote_handle),
  m_remote_handle_size(remote_handle_size)
{
    if ((m_remote_handle != nullptr) || (m_remote_handle_size != 0U))
    {
        CHECK(m_remote_handle && m_remote_handle_size);
    }
}

ucp_mem_h MemoryBlock::local_handle() const
{
    return m_local_handle;
}

void* MemoryBlock::remote_handle() const
{
    return m_remote_handle;
}

std::size_t MemoryBlock::remote_handle_size() const
{
    return m_remote_handle_size;
}

std::string MemoryBlock::packed_remote_keys() const
{
    std::string keys;
    keys.resize(m_remote_handle_size);
    std::memcpy(keys.data(), m_remote_handle, m_remote_handle_size);
    return keys;
}

}  // namespace srf::internal::ucx

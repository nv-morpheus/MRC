/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/memory/transient_pool.hpp"

#include "srf/memory/resources/memory_resource.hpp"

namespace srf::internal::memory {

TransientBuffer::TransientBuffer(void* addr, std::size_t bytes, srf::data::SharedReusable<srf::memory::buffer> buffer) :
  m_addr(addr),
  m_bytes(bytes),
  m_buffer(std::move(buffer))
{}

void* TransientBuffer::data()
{
    return m_addr;
}

std::size_t TransientBuffer::bytes() const
{
    return m_bytes;
}

void TransientBuffer::release()
{
    m_addr  = nullptr;
    m_bytes = 0;
    m_buffer.release();
}

TransientPool::TransientPool(std::size_t block_size,
                             std::size_t block_count,
                             std::size_t capacity,
                             std::shared_ptr<srf::memory::memory_resource> mr) :
  m_block_size(block_size),
  m_pool(srf::data::ReusablePool<srf::memory::buffer>::create(capacity))
{
    CHECK(m_pool);
    CHECK_LT(block_count, capacity);
    for (int i = 0; i < block_count; i++)
    {
        m_pool->emplace(block_size, mr);
    }
}

TransientBuffer TransientPool::await_buffer(std::size_t bytes)
{
    if (bytes > m_block_size)  // todo(#54) [[unlikely]]
    {
        LOG(ERROR) << "requesting allocation larger than max_size";
        throw std::bad_alloc{};
    }

    if (m_remaining < bytes)
    {
        auto buffer = m_pool->await_item();
        m_addr      = static_cast<std::byte*>(buffer->data());
        m_remaining = buffer->bytes();
        m_buffer    = std::move(buffer);
    }

    // align + bytes
    void* addr = m_addr;
    m_addr += bytes;
    m_remaining -= bytes;

    return TransientBuffer(addr, bytes, m_buffer);
}

}  // namespace srf::internal::memory

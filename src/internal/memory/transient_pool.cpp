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

#include "mrc/memory/resources/memory_resource.hpp"

#include <ostream>

#define MRC_DEBUG 1

namespace mrc::internal::memory {

TransientBuffer::TransientBuffer(void* addr, std::size_t bytes, mrc::data::SharedReusable<mrc::memory::buffer> buffer) :
  m_addr(addr),
  m_bytes(bytes),
  m_buffer(std::move(buffer))
{}

TransientBuffer::TransientBuffer(void* addr, std::size_t bytes, const TransientBuffer& buffer) :
  m_addr(addr),
  m_bytes(bytes),
  m_buffer(buffer.m_buffer)
{
    auto* c = static_cast<std::byte*>(addr);
    auto* b = static_cast<std::byte*>(const_cast<void*>(buffer.data()));
    CHECK_GE(c, b);
    c += bytes;
    b += buffer.bytes();
    CHECK_LE(c, b);
}

TransientBuffer::~TransientBuffer()
{
    release();
}

TransientBuffer::TransientBuffer(TransientBuffer&& other) noexcept :
  m_addr(std::exchange(other.m_addr, nullptr)),
  m_bytes(std::exchange(other.m_bytes, 0UL)),
  m_buffer(std::move(other.m_buffer))
{}

TransientBuffer& TransientBuffer::operator=(TransientBuffer&& other) noexcept
{
    m_addr   = std::exchange(other.m_addr, nullptr);
    m_bytes  = std::exchange(other.m_bytes, 0UL);
    m_buffer = std::move(other.m_buffer);
    return *this;
}

void* TransientBuffer::data()
{
    return m_addr;
}

const void* TransientBuffer::data() const
{
    return m_addr;
}

std::size_t TransientBuffer::bytes() const
{
    return m_bytes;
}

void TransientBuffer::release()
{
    if (m_addr != nullptr)
    {
        m_addr  = nullptr;
        m_bytes = 0;
        m_buffer.release();
    }
}

TransientPool::TransientPool(std::size_t block_size,
                             std::size_t block_count,
                             std::shared_ptr<mrc::memory::memory_resource> mr,
                             std::size_t capacity) :
  m_block_size(block_size),
  m_pool(mrc::data::ReusablePool<mrc::memory::buffer>::create(capacity))
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

    return {addr, bytes, m_buffer};
}

}  // namespace mrc::internal::memory

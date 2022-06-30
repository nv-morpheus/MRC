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

#pragma once

#include "srf/data/reusable_pool.hpp"
#include "srf/memory/buffer.hpp"
#include "srf/memory/resources/memory_resource.hpp"
#include "srf/utils/macros.hpp"

#include <glog/logging.h>

#include <cstddef>
#include <new>
#include <utility>

namespace srf::internal::memory {

/**
 * @brief A short-lived buffer based on a portion of a data::SharedResable<memory::buffer>
 *
 * @see TransientPool for more details.
 */

class TransientBuffer
{
  public:
    TransientBuffer(void* addr, std::size_t bytes, srf::data::SharedReusable<srf::memory::buffer> buffer);
    ~TransientBuffer();

    DELETE_COPYABILITY(TransientBuffer);
    DEFAULT_MOVEABILITY(TransientBuffer);

    void* data();
    std::size_t bytes() const;

    void release();

  private:
    void* m_addr;
    std::size_t m_bytes;
    srf::data::SharedReusable<srf::memory::buffer> m_buffer;
};

/**
 * @brief A short-lived object T allocated into the static storage of a TransientBuffer.
 *
 * @see TransientPool and TransientBuffer for more details.
 */

template <typename T>
class Transient final : private TransientBuffer
{
  public:
    Transient(TransientBuffer&& buffer) : TransientBuffer(std::move(buffer))
    {
        CHECK_LT(sizeof(T), bytes());
        void* addr       = data();
        std::size_t size = bytes();
        addr             = std::align(alignof(T), sizeof(T), addr, size);
        CHECK(addr);
        m_data = new (addr) T;
    }

    ~Transient()
    {
        release();
    }

    T& operator*()
    {
        CHECK(m_data);
        return *m_data;
    }

    T* operator->()
    {
        CHECK(m_data);
        return m_data;
    }

    void release()
    {
        if (m_data != nullptr)
        {
            m_data->~T();
            m_data = nullptr;
            TransientBuffer::release();
        }
    }

  private:
    T* m_data;
};

/**
 * @brief ReusablePool of memory::buffers that are used as reusable reference-counted monotonic memory resources
 *
 * TransientPool is a ReusablePool of memory::buffers from which smaller buffers are allocated similar to a monotonic
 * memory resource, i.e. pointer pushing stack; however the TransientBuffer or Transient<T> object pull from the pool
 * hold a SharedReusable<memory::buffer> which keeps the entire monotonic stack from returning to the resuable pool
 * until all objects created on a given stack are deallocated.
 *
 * Allocation of Transisent object should be incredibly fast; even faster than the Reusable/SharedResuable on which they
 * are based, since a single Reusable<memory::buffer> might back 10s-1000s of allocations dependending on size.
 *
 * It is critical that all Transient object allocated from a pool have similar life cycles
 */
class TransientPool
{
  public:
    TransientPool(std::size_t block_size,
                  std::size_t block_count,
                  std::size_t capacity,
                  std::shared_ptr<srf::memory::memory_resource> mr);

    TransientBuffer await_buffer(std::size_t bytes);

    template <typename T>
    Transient<T> await_object()
    {
        auto buffer = await_buffer(sizeof(T) + alignof(T));
        return Transient<T>(std::move(buffer));
    }

  private:
    const std::size_t m_block_size;
    const std::shared_ptr<srf::data::ReusablePool<srf::memory::buffer>> m_pool;
    std::byte* m_addr{nullptr};
    std::size_t m_remaining{0};
    srf::data::SharedReusable<srf::memory::buffer> m_buffer;
};

}  // namespace srf::internal::memory

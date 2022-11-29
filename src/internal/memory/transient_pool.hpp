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

#include "mrc/data/reusable_pool.hpp"
#include "mrc/memory/buffer.hpp"
#include "mrc/memory/resources/memory_resource.hpp"
#include "mrc/utils/macros.hpp"

#include <glog/logging.h>

#include <cstddef>
#include <memory>
#include <new>
#include <utility>

namespace mrc::internal::memory {

/**
 * @brief A short-lived buffer based on a portion of a data::SharedResable<memory::buffer>
 *
 * @see TransientPool for more details.
 */

class TransientBuffer
{
  public:
    /**
     * @brief Construct a new Transient Buffer object
     *
     * @param addr - starting address of the buffer
     * @param bytes - number of bytes allocated to this buffer starting at addr
     * @param buffer - reference counted holder to the SharedReusable, this is held for reference counting only
     */
    TransientBuffer(void* addr, std::size_t bytes, mrc::data::SharedReusable<mrc::memory::buffer> buffer);
    TransientBuffer(void* addr, std::size_t bytes, const TransientBuffer& buffer);

    TransientBuffer() = default;
    ~TransientBuffer();

    TransientBuffer(TransientBuffer&& other) noexcept;
    TransientBuffer& operator=(TransientBuffer&& other) noexcept;

    DELETE_COPYABILITY(TransientBuffer);

    /**
     * @brief Starting address of the TransientBuffer
     */
    void* data();
    const void* data() const;

    /**
     * @brief Capacity of TransientBuffer in number of bytes extending from data().
     */
    std::size_t bytes() const;

    /**
     * @brief Release the buffer and nullifies the starting address and resets the capacity in bytes to 0.
     */
    void release();

  private:
    void* m_addr{nullptr};
    std::size_t m_bytes{0};
    mrc::data::SharedReusable<mrc::memory::buffer> m_buffer;
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
    template <typename... ArgsT>
    Transient(TransientBuffer&& buffer, ArgsT&&... args) : TransientBuffer(std::move(buffer))
    {
        CHECK_LE(sizeof(T) + alignof(T), bytes());
        void* addr       = data();
        std::size_t size = bytes();
        CHECK(std::align(alignof(T), sizeof(T), addr, size));
        m_data = new (addr) T(std::forward<ArgsT>(args)...);
    }

    Transient(Transient&& other) noexcept :
      TransientBuffer(std::move(other)),
      m_data(std::exchange(other.m_data, nullptr))
    {}

    Transient& operator=(Transient&& other) noexcept
    {
        TransientBuffer::operator=(std::move(other));
        m_data                   = std::exchange(other.m_data, nullptr);
        return *this;
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

    operator bool() const
    {
        return (static_cast<bool>(m_data));
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
                  std::shared_ptr<mrc::memory::memory_resource> mr,
                  std::size_t capacity = 64);

    /**
     * @brief Acquire a TransientBuffer of size bytes.
     *
     * Acquiring this object may block and yield the calling fiber.
     *
     * @param bytes
     * @return TransientBuffer
     */
    TransientBuffer await_buffer(std::size_t bytes);

    /**
     * @brief Acquire a Transient<T> constructed from a TransientBuffer.
     *
     * Acquiring this object may block and yield the calling fiber.
     *
     * @tparam T
     * @return Transient<T>
     */
    template <typename T, typename... ArgsT>
    Transient<T> await_object(ArgsT&&... args)
    {
        auto buffer = await_buffer(sizeof(T) + alignof(T));
        return Transient<T>(std::move(buffer), std::forward<ArgsT>(args)...);
    }

  private:
    const std::size_t m_block_size;
    const std::shared_ptr<mrc::data::ReusablePool<mrc::memory::buffer>> m_pool;
    std::byte* m_addr{nullptr};
    std::size_t m_remaining{0};
    mrc::data::SharedReusable<mrc::memory::buffer> m_buffer;
};

}  // namespace mrc::internal::memory

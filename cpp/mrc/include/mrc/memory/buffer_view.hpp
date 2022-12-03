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

#include "mrc/memory/buffer.hpp"
#include "mrc/memory/memory_kind.hpp"

#include <cstddef>

namespace mrc::memory {

class buffer_view;

/**
 * @brief Generic container for holding the address to a const memory buffer_view.
 *
 * Minimum description of a buffer_view of contiguous memory / bytes with a specified type of memory.
 *
 * const_buffer_view is a copyable and does not own the described memory buffer_view.
 */
class const_buffer_view  // NOLINT
{
  public:
    const_buffer_view()          = default;
    virtual ~const_buffer_view() = default;

    // todo(clang-format-15)
    // clang-format off
    const_buffer_view(const const_buffer_view& other)      = default;
    const_buffer_view& operator=(const const_buffer_view&) = default;
    // clang-format on

    const_buffer_view(const_buffer_view&& other) noexcept;
    const_buffer_view& operator=(const_buffer_view&& other) noexcept;

    /**
     * @brief Construct a new const_buffer_view object from a raw pointer and details
     *
     * @param data
     * @param bytes
     * @param kind
     */
    const_buffer_view(void* data, std::size_t bytes, memory_kind kind);

    /**
     * @brief Construct a new const_buffer_view object from a const raw pointer and details
     *
     * @param data
     * @param bytes
     * @param kind
     */
    const_buffer_view(const void* data, std::size_t bytes, memory_kind kind);

    /**
     * @brief Construct a const_buffer_view from a buffer
     *
     * @tparam Properties
     * @param buffer
     */
    const_buffer_view(const buffer& buffer);

    /**
     * @brief Constant pointer to the start of the memory buffer_view
     *
     * @return const void*
     */
    const void* data() const;

    /**
     * @brief Number of bytes, i.e. the capacity of the memory buffer_view
     *
     * @return std::size_t
     */
    std::size_t bytes() const;

    /**
     * @brief Type of memory described by the buffer_view
     *
     * @return memory_kind
     */
    memory_kind kind() const;

    /**
     * @brief Returns true if the memory buffer_view is empty
     *
     * A memory buffer_view is empty if it does not hold a backing storage object or
     * if that backing storage object points to nullptr and is of size zero.
     *
     * @return true
     * @return false
     */
    bool empty() const;

    /**
     * @brief bool operator, returns true if the view is backed by some storage whose
     *        buffer_view has a valid starting address and a specified size.
     *
     * @return true
     * @return false
     */
    operator bool() const;

  private:
    void* m_data{nullptr};
    std::size_t m_bytes{0UL};
    memory_kind m_kind{memory_kind::none};

    friend buffer_view;
};

/**
 * @brief Generic container for holding the address to a mutable memory buffer_view.
 *
 * Minimum description of a buffer_view of contiguous memory / bytes with a specified type of memory.
 *
 * buffer_view is a copyable and does not own the described memory buffer_view.
 */
class buffer_view : public const_buffer_view
{
  public:
    using const_buffer_view::const_buffer_view;

    /**
     * @brief
     *
     * @return void*
     */
    void* data();
};

}  // namespace mrc::memory

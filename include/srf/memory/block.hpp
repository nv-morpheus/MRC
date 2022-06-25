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

#include "srf/memory/buffer.hpp"
#include "srf/memory/memory_kind.hpp"

#include <glog/logging.h>

#include <cstddef>  // for size_t

namespace srf::memory {

class block;

/**
 * @brief Generic container for holding the address to a const memory block.
 *
 * Minimum description of a block of contiguous memory / bytes with a specified type of memory.
 *
 * const_block is a copyable and does not own the described memory block.
 */
class const_block  // NOLINT
{
  public:
    const_block()          = default;
    virtual ~const_block() = default;

    /**
     * @brief Construct a new const_block object from a raw pointer and details
     *
     * @param data
     * @param bytes
     * @param kind
     */
    const_block(void* data, std::size_t bytes, memory_kind kind);

    /**
     * @brief Construct a new const_block object from a const raw pointer and details
     *
     * @param data
     * @param bytes
     * @param kind
     */
    const_block(const void* data, std::size_t bytes, memory_kind kind);

    /**
     * @brief Construct a const_block from a blob
     *
     * @param blob
     */
    // const_block(const blob& blob) : m_data(const_cast<void*>(blob.data())), m_bytes(blob.bytes()),
    // m_kind(blob.kind())
    // {
    //     CHECK(operator bool());
    // }

    /**
     * @brief Construct a const_block from a buffer
     *
     * @tparam Properties
     * @param buffer
     */
    const_block(const buffer& buffer) :
      m_data(const_cast<void*>(buffer.data())),
      m_bytes(buffer.bytes()),
      m_kind(buffer.kind())
    {
        CHECK(operator bool());
    }

    /**
     * @brief Constant pointer to the start of the memory block
     *
     * @return const void*
     */
    const void* data() const;

    /**
     * @brief Number of bytes, i.e. the capacity of the memory block
     *
     * @return std::size_t
     */
    std::size_t bytes() const;

    /**
     * @brief Type of memory described by the block
     *
     * @return memory_kind
     */
    memory_kind kind() const;

    /**
     * @brief Returns true if the memory block is empty
     *
     * A memory block is empty if it does not hold a backing storage object or
     * if that backing storage object points to nullptr and is of size zero.
     *
     * @return true
     * @return false
     */
    bool empty() const;

    /**
     * @brief bool operator, returns true if the view is backed by some storage whose
     *        block has a valid starting address and a specified size.
     *
     * @return true
     * @return false
     */
    operator bool() const;

  private:
    void* m_data{nullptr};
    std::size_t m_bytes{0UL};
    memory_kind m_kind{memory_kind::none};

    friend block;
};

/**
 * @brief Generic container for holding the address to a mutable memory block.
 *
 * Minimum description of a block of contiguous memory / bytes with a specified type of memory.
 *
 * block is a copyable and does not own the described memory block.
 */
class block : public const_block
{
  public:
    using const_block::const_block;

    /**
     * @brief
     *
     * @return void*
     */
    void* data();
};

}  // namespace srf::memory

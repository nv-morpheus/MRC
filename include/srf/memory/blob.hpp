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

#include "srf/memory/blob_storage.hpp"
#include "srf/memory/memory_kind.hpp"

#include <cstddef>  // for size_t
#include <memory>
#include <type_traits>  // for enable_if_t & remove_reference_t

namespace srf::memory {

/**
 * @brief Generic container for holding bytes or blobs, a.k.a Binary Large Object
 *
 * Minimum description of a block of contiguous memory / bytes with a specified type of memory.
 *
 * blob is a copyable, internally reference-counted object that describes a single contiguous chunck of memory.
 * Assignments and copies are new blob objects with a shared reference count.
 *
 * Ownership is required - either by an explicit move or a capture of a shared_ptr. The held memory will be released
 * when the last blob and/or the last shared_ptr referencing the memory block have been released.
 */
class blob
{
  public:
    blob();
    virtual ~blob();

    blob(blob&&) noexcept = default;
    blob& operator=(blob&&) noexcept = default;

    blob(const blob&) = default;
    blob& operator=(const blob&) = default;

    /**
     * @brief Construct a new buffer view object from a raw pointer and details
     *
     * @param data
     * @param bytes
     * @param kind
     */
    // template<typename... Properties>
    // blob(void* data, std::size_t bytes, memory_kind_type kind, resource_view<Properties...> view);

    /**
     * @brief Construct a new blob from a shared_ptr to the internal interface
     *
     * @param view
     */
    explicit blob(std::shared_ptr<IBlobStorage> view);

    /**
     * @brief Construct a new buffer view object from a StorageT object whose lifecycle can be managed by the StorageT
     * object for which the view takes ownership.
     *
     * @note BlobStorage can be extended via template specialization to hold user defined objects
     *
     * @tparam StorageT
     */
    template <typename StorageT,
              typename = std::enable_if_t<!std::is_base_of_v<blob, std::remove_reference_t<StorageT>>>>
    blob(StorageT&& storage) : m_storage(std::make_shared<BlobStorage<StorageT>>(std::move(storage)))
    {}

    /**
     * @brief Pointer to the start of the memory block described by blob
     *
     * @return void*
     */
    void* data();

    /**
     * @brief Constant pointer to the start of the memory block descibed by blob
     *
     * @return const void*
     */
    const void* data() const;

    /**
     * @brief Number of bytes, i.e. the capacity of the memory block described by blob
     *
     * @return std::size_t
     */
    std::size_t bytes() const;

    /**
     * @brief Type of memory described by the blob
     *
     * @return memory_kind_type
     */
    memory_kind_type kind() const;

    /**
     * @brief Value of the internal reference count to the object backing the blob
     *
     * @return auto
     */
    auto use_count() const
    {
        return m_storage.use_count();
    }

    /**
     * @brief Returns true if the blob is empty
     *
     * A buffer view is empty if it does not hold a backing storage object or
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

    /**
     * @brief allocate a new blob
     *
     * @param bytes
     * @return blob
     */
    blob allocate(std::size_t bytes) const;

  private:
    std::shared_ptr<IBlobStorage> m_storage{nullptr};
};

}  // namespace srf::memory

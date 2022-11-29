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

#include "pymrc/utilities/object_cache.hpp"

#include <pybind11/pytypes.h>

#include <cstddef>

namespace mrc::pymrc {
/**
 * @brief Wrapper around the multiprocess.shared_memory class
 **/
#pragma GCC visibility push(default)
class PythonSharedMemoryInterface
{
  public:
    ~PythonSharedMemoryInterface();

    PythonSharedMemoryInterface();

    /**
     * @brief allocate a new block of shared memory
     * @param sz_bytes
     */
    void allocate(std::size_t sz_bytes);

    /**
     * @brief Attach to an existing shared memory block using a descriptor object
     * @param block_id python string identifying the block id.
     */
    void attach(pybind11::object block_id);

    /**
     * @brief Close the shared memory object, shared memory still exists.
     */
    void close();

    /**
     * @brief Utility function that wraps, close, unlink, and unsetting the object
     */
    void free();

    /**
     * @brief Get the bytes object associated with this shared memory
     * @return
     */
    pybind11::bytes get_bytes() const;

    /**
     * @brief Get the memoryview associated with the shared memory object.
     * @return memoryview
     */
    pybind11::memoryview get_memoryview() const;

    /**
     * @brief Get the block id of the current shared memory object
     * @return
     */
    pybind11::object block_id() const;

    /**
     * @brief Set the bytes content of the shared memory object -- must be the same size.
     * @param bytes
     */
    void set(pybind11::bytes bytes);

    /**
     * @brief get the size, in bytes, of the shared memory object.
     * @return
     */
    pybind11::object size_bytes() const;

    /**
     * @brief release the shared memory block.
     */
    void unlink();

  private:
    PythonObjectCache& m_pycache;

    pybind11::object m_shmem_interface{};
    pybind11::object m_shmem{pybind11::none()};
};

/**
 *
 * @param shmem_interface Interface object representing the shared memory that we're building a descriptor for.
 * @param flag_is_shared
 * @return Descriptor object for retrieving the shared memory.
 */
pybind11::object build_shmem_descriptor(const PythonSharedMemoryInterface& shmem_interface,
                                        bool flag_is_shared = false);

#pragma GCC visibility pop
}  // namespace mrc::pymrc

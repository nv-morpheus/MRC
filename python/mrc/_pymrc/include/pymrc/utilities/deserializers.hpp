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

#include <pybind11/buffer_info.h>
#include <pybind11/pytypes.h>

#include <cstddef>

namespace mrc::pymrc {
#pragma GCC visibility push(default)
struct Deserializer
{
    /**
     * @brief Deserialize from various formats
     * @param bytes
     * @return
     */
    static pybind11::object deserialize(pybind11::bytes bytes);
    static pybind11::object deserialize(pybind11::buffer_info& buffer_info);
    static pybind11::object deserialize(const char* bytes, std::size_t count);

    /**
     * @brief Given a pyMRC shmem descriptor, attempt to retrieve the object information from shared memory
     * and unpickle it.
     * @param descriptor
     * @return
     */
    static pybind11::object load_from_shared_memory(pybind11::object descriptor);
};
#pragma GCC visibility pop
}  // namespace mrc::pymrc

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

#include <pybind11/pytypes.h>

#include <cstddef>
#include <tuple>

namespace mrc::pymrc {
#pragma GCC visibility push(default)

struct Serializer
{
    /**
     *
     * @param obj pybind11 object to serialize
     * @param use_shmem flag indicating whether or not we should put the serialized object into shared memory.
     * @return
     */
    static std::tuple<char*, std::size_t> serialize(pybind11::object obj,
                                                    bool use_shmem         = false,
                                                    bool return_raw_buffer = false);
    static pybind11::object persist_to_shared_memory(pybind11::object obj);
};
#pragma GCC visibility pop
}  // namespace mrc::pymrc

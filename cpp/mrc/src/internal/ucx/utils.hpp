/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/ucx/common.hpp"

#include "mrc/memory/memory_kind.hpp"

#include <ucs/memory/memory_type.h>

namespace mrc::ucx {

inline ucs_memory_type_t to_ucs_memory_type(memory::memory_kind kind)
{
    switch (kind)
    {
    case memory::memory_kind::none:
        return UCS_MEMORY_TYPE_UNKNOWN;
    case memory::memory_kind::host:
    case memory::memory_kind::pinned:
        return UCS_MEMORY_TYPE_HOST;
    case memory::memory_kind::device:
        return UCS_MEMORY_TYPE_CUDA;
    case memory::memory_kind::managed:
        return UCS_MEMORY_TYPE_CUDA_MANAGED;
    default:
        throw std::runtime_error("Unknown memory kind");
    }
}

}  // namespace mrc::ucx

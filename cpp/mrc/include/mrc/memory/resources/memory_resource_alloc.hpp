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

#include "mrc/memory/memory_kind.hpp"
#include "mrc/memory/resources/memory_resource.hpp"
#include "mrc/memory/resources/device/cuda_malloc_resource.hpp"
#include "mrc/memory/resources/host/malloc_memory_resource.hpp"
#include "mrc/memory/resources/host/pinned_memory_resource.hpp"

namespace mrc::memory {

static std::shared_ptr<memory_resource> fetch_memory_resource_instance(memory_kind kind)
{
    switch (kind)
    {
    case memory_kind::none:
        break;

    case memory_kind::host:
        static std::shared_ptr<malloc_memory_resource> host = malloc_memory_resource::instance();
        return host;
        break;

    case memory_kind::pinned:
        static std::shared_ptr<pinned_memory_resource> pinned = pinned_memory_resource::instance();
        return pinned;
        break;

    case memory_kind::device:
        static std::shared_ptr<cuda_malloc_resource2> device = cuda_malloc_resource2::instance();
        return device;
        break;

    case memory_kind::managed:
        break;
    }

    LOG(FATAL) << "memory_kind is either none or unknown";
    return nullptr;
}

}  // namespace mrc::memory

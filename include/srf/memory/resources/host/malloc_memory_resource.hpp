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

#include "srf/memory/resources/memory_resource.hpp"

namespace srf::memory {

class malloc_memory_resource : public memory_resource<cuda::memory_kind::host>
{
    void* do_allocate(std::size_t bytes, std::size_t /*__alignment*/) final
    {
        return std::malloc(bytes);
    }

    void do_deallocate(void* ptr, std::size_t /*__bytes*/, std::size_t /*__alignment*/) final
    {
        std::free(ptr);
    }

    memory_kind_type do_kind() const final
    {
        return memory_kind_type::host;
    }

  public:
    malloc_memory_resource() : memory_resource("malloc")
    {
        // todo(ryan) - check for the presence of cuda devices - fail if found
    }
    ~malloc_memory_resource() override = default;
};

// will perform a runtime check to ensure no cuda devices are present
// this memory resource will provide extra properties such as device accessible
// this allows for using the greatest common demoninator of properties across
// srf. when no devices are accessible, the device_accessible flag is meaningless

class malloc_without_device_resources : public memory_resource<cuda::memory_kind::pinned>
{
    void* do_allocate(std::size_t bytes, std::size_t /*__alignment*/) final
    {
        return std::malloc(bytes);
    }

    void do_deallocate(void* ptr, std::size_t /*__bytes*/, std::size_t /*__alignment*/) final
    {
        std::free(ptr);
    }

    memory_kind_type do_kind() const final
    {
        return memory_kind_type::host;
    }

  public:
    malloc_without_device_resources() : memory_resource("malloc_without_device")
    {
        // todo(ryan) - check for the presence of cuda devices - fail if found
    }
    ~malloc_without_device_resources() override = default;
};

}  // namespace srf::memory

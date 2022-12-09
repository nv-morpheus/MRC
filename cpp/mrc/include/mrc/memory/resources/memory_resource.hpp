/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <vector>

namespace mrc::memory {

struct memory_resource
{
    virtual ~memory_resource() = default;

    void* allocate(std::size_t bytes)
    {
        return do_allocate(bytes);
    }

    void deallocate(void* ptr, std::size_t bytes)
    {
        if (ptr != nullptr)
        {
            do_deallocate(ptr, bytes);
        }
    }

    memory_kind kind() const
    {
        return do_kind();
    }

  private:
    virtual void* do_allocate(std::size_t bytes)             = 0;
    virtual void do_deallocate(void* ptr, std::size_t bytes) = 0;
    virtual memory_kind do_kind() const                      = 0;
};

}  // namespace mrc::memory

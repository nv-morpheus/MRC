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

#include "mrc/options/resources.hpp"

#include "mrc/memory/literals.hpp"

namespace mrc {

using namespace memory::literals;

// Resource Options

MemoryPoolOptions::MemoryPoolOptions(std::size_t block_size, std::size_t max_aggregate_bytes) :
  m_block_size(block_size),
  m_max_aggregate_bytes(max_aggregate_bytes)
{}

MemoryPoolOptions& ResourceOptions::host_memory_pool()
{
    return m_host_memory_pool;
}

MemoryPoolOptions& ResourceOptions::device_memory_pool()
{
    return m_device_memory_pool;
}

const MemoryPoolOptions& ResourceOptions::host_memory_pool() const
{
    return m_host_memory_pool;
}

const MemoryPoolOptions& ResourceOptions::device_memory_pool() const
{
    return m_device_memory_pool;
}

ResourceOptions::ResourceOptions() : m_host_memory_pool(128_MiB, 1_GiB), m_device_memory_pool(128_MiB, 1_GiB) {}

ResourceOptions& ResourceOptions::enable_host_memory_pool(bool value)
{
    m_enable_host_memory_pool = value;
    return *this;
}

ResourceOptions& ResourceOptions::enable_device_memory_pool(bool value)
{
    m_enable_device_memory_pool = value;
    return *this;
}

bool ResourceOptions::enable_host_memory_pool() const
{
    return m_enable_host_memory_pool;
}

bool ResourceOptions::enable_device_memory_pool() const
{
    return m_enable_device_memory_pool;
}

}  // namespace mrc

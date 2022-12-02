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

#include <cstddef>  // for size_t

namespace mrc {

class MemoryPoolOptions
{
  public:
    MemoryPoolOptions(std::size_t block_size, std::size_t max_aggregate_bytes);

    MemoryPoolOptions& block_size(std::size_t size)
    {
        m_block_size = size;
        return *this;
    }
    MemoryPoolOptions& max_aggregate_bytes(std::size_t count)
    {
        m_max_aggregate_bytes = count;
        return *this;
    }

    [[nodiscard]] std::size_t block_size() const
    {
        return m_block_size;
    }
    [[nodiscard]] std::size_t max_aggreate_bytes() const
    {
        return m_max_aggregate_bytes;
    }

  private:
    std::size_t m_block_size;
    std::size_t m_max_aggregate_bytes;
};

class ResourceOptions
{
  public:
    ResourceOptions();

    /**
     * @brief enable/disable host memory pool
     *
     * @return ResourceOptions&
     */
    ResourceOptions& enable_host_memory_pool(bool);

    /**
     * @brief enable/disable device memory pool
     *
     * @return ResourceOptions&
     */
    ResourceOptions& enable_device_memory_pool(bool);

    /**
     * @brief respect the process affinity as launch by the system (default: true)
     **/
    MemoryPoolOptions& host_memory_pool();

    /**
     * @brief user specified subset of logcial cpus on which to run
     **/
    MemoryPoolOptions& device_memory_pool();

    bool enable_host_memory_pool() const;
    bool enable_device_memory_pool() const;
    const MemoryPoolOptions& host_memory_pool() const;
    const MemoryPoolOptions& device_memory_pool() const;

  private:
    bool m_enable_host_memory_pool{false};
    bool m_enable_device_memory_pool{false};
    MemoryPoolOptions m_host_memory_pool;
    MemoryPoolOptions m_device_memory_pool;
};

}  // namespace mrc

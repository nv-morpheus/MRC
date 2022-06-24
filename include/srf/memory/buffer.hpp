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

#include <srf/memory/adaptors.hpp>
#include <srf/memory/resources/memory_resource.hpp>

#include <cuda/memory_resource>
#include <rmm/device_buffer.hpp>

#include <cstddef>
#include <utility>

namespace srf::memory {

class buffer
{
  public:
    buffer() = default;
    buffer(std::size_t bytes, memory_resource* mr) :
      m_mr(mr),
      m_bytes(std::move(bytes)),
      m_buffer(m_mr->allocate(m_bytes))
    {}

    virtual ~buffer()
    {
        release();
    }

    buffer(const buffer&) = delete;
    buffer& operator=(const buffer&) = delete;

    buffer(buffer&& other) noexcept :
      m_mr(std::exchange(other.m_mr, nullptr)),
      m_bytes(std::exchange(other.m_bytes, 0)),
      m_buffer(std::exchange(other.m_buffer, nullptr))
    {}

    buffer& operator=(buffer&& other) noexcept
    {
        m_mr     = std::exchange(other.m_mr, nullptr);
        m_bytes  = std::exchange(other.m_bytes, 0);
        m_buffer = std::exchange(other.m_buffer, nullptr);
        return *this;
    }

    void release()
    {
        if (m_buffer != nullptr)
        {
            m_mr->deallocate(m_buffer, m_bytes);
            m_buffer = nullptr;
            m_bytes  = 0;
        }
    }

    void* data() noexcept
    {
        return m_buffer;
    }
    const void* data() const noexcept
    {
        return m_buffer;
    }

    std::size_t bytes() const noexcept
    {
        return m_bytes;
    }

    memory_kind kind() const noexcept
    {
        return m_mr->kind();
    }

    bool empty() const
    {
        return not bool(*this);
    }

    operator bool() const
    {
        return (m_buffer != nullptr) && (m_bytes != 0U);
    }

  private:
    memory_resource* m_mr;
    std::size_t m_bytes{0};
    void* m_buffer{nullptr};
};

}  // namespace srf::memory

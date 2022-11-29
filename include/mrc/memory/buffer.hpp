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

#include "mrc/memory/adaptors.hpp"
#include "mrc/memory/resources/memory_resource.hpp"
#include "mrc/utils/bytes_to_string.hpp"

#include <cstddef>
#include <utility>

namespace mrc::memory {

class buffer
{
  public:
    buffer() = default;

    explicit buffer(std::size_t bytes, std::shared_ptr<memory_resource> mr) :
      m_mr(std::move(mr)),
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
            CHECK(m_mr);
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
        DCHECK(m_mr);
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

    bool contains(const void* ptr) const
    {
        const auto* p = static_cast<const std::byte*>(ptr);
        auto* s       = static_cast<std::byte*>(m_buffer);
        auto* e       = s + m_bytes;
        return (m_buffer != nullptr && s <= p && p < e);
    }

  private:
    std::shared_ptr<memory_resource> m_mr;
    std::size_t m_bytes{0};
    void* m_buffer{nullptr};

    friend std::ostream& operator<<(std::ostream& os, const buffer& buffer);
};

inline std::ostream& operator<<(std::ostream& os, const mrc::memory::buffer& buffer)
{
    os << "[memory::buffer " << buffer.data() << "; bytes=" << mrc::bytes_to_string(buffer.bytes())
       << "; kind= " << mrc::memory::kind_string(buffer.kind()) << "]";
    return os;
}

}  // namespace mrc::memory

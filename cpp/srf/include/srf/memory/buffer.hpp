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
#include <srf/memory/resource_view.hpp>
#include <srf/memory/resources/memory_resource.hpp>

#include <cuda/memory_resource>
#include <rmm/device_buffer.hpp>

#include <cstddef>
#include <utility>

namespace srf::memory {

template <typename... Properties>
class buffer
{
  public:
    template <typename Property>
    struct contains : std::bool_constant<(std::is_same<Property, Properties>{} || ...)>
    {};

    using view_type = resource_view<Properties...>;

    buffer() = default;
    buffer(std::size_t bytes, view_type view) :
      m_view(std::move(view)),
      m_bytes(std::move(bytes)),
      m_buffer(m_view.allocate(m_bytes))
    {}
    virtual ~buffer()
    {
        release();
    }

    buffer(const buffer&) = delete;
    buffer& operator=(const buffer&) = delete;

    buffer(buffer&& other) noexcept :
      m_view(std::move(other.m_view)),
      m_bytes(std::exchange(other.m_bytes, 0)),
      m_buffer(std::exchange(other.m_buffer, nullptr))
    {}

    buffer& operator=(buffer&& other) noexcept
    {
        m_view   = std::move(other.m_view);
        m_bytes  = std::exchange(other.m_bytes, 0);
        m_buffer = std::exchange(other.m_buffer, nullptr);
        return *this;
    }

    void release()
    {
        if (m_buffer != nullptr)
        {
            m_view.deallocate(m_buffer, m_bytes, alignof(std::max_align_t));
            m_buffer = nullptr;
            m_bytes  = 0;
        }
    }

    template <typename P>
    static constexpr bool has_property()
    {
        return std::bool_constant<(std::is_same<P, Properties>{} || ...)>::value;
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

    memory_kind_type kind() const noexcept
    {
        return m_view.kind();
    }

    const view_type& view() const noexcept
    {
        return m_view;
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
    view_type m_view;
    std::size_t m_bytes{0};
    void* m_buffer{nullptr};
};

}  // namespace srf::memory

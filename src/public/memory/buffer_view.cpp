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

#include "mrc/memory/buffer_view.hpp"

#include "mrc/memory/memory_kind.hpp"

#include <glog/logging.h>

#include <utility>

namespace mrc::memory {

// For the move constructor, reset the values in `other` to the defaults
const_buffer_view::const_buffer_view(const_buffer_view&& other) noexcept :
  m_data(std::exchange(other.m_data, nullptr)),
  m_bytes(std::exchange(other.m_bytes, 0UL)),
  m_kind(std::exchange(other.m_kind, memory_kind::none))
{}

const_buffer_view& const_buffer_view::operator=(const_buffer_view&& other) noexcept
{
    std::swap(m_data, other.m_data);
    std::swap(m_bytes, other.m_bytes);
    std::swap(m_kind, other.m_kind);

    return *this;
}

const_buffer_view::const_buffer_view(void* data, std::size_t bytes, memory_kind kind) :
  m_data(data),
  m_bytes(bytes),
  m_kind(kind)
{}

const_buffer_view::const_buffer_view(const void* data, std::size_t bytes, memory_kind kind) :
  m_data(const_cast<void*>(data)),
  m_bytes(bytes),
  m_kind(kind)
{}

const_buffer_view::const_buffer_view(const buffer& buffer) :
  m_data(const_cast<void*>(buffer.data())),
  m_bytes(buffer.bytes()),
  m_kind(buffer.kind())
{
    CHECK(operator bool());
}

const void* const_buffer_view::data() const
{
    return m_data;
}

std::size_t const_buffer_view::bytes() const
{
    return m_bytes;
}

memory_kind const_buffer_view::kind() const
{
    return m_kind;
}

bool const_buffer_view::empty() const
{
    return not bool(*this);
}

const_buffer_view::operator bool() const
{
    return (m_data != nullptr) && (m_bytes != 0U) && (m_kind != memory_kind::none);
}

void* buffer_view::data()
{
    return m_data;
}

}  // namespace mrc::memory

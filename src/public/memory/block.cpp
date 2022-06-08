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

#include <srf/memory/block.hpp>
#include <srf/memory/memory_kind.hpp>

namespace srf::memory {

const_block::const_block(void* data, std::size_t bytes, memory_kind_type kind) :
  m_data(data),
  m_bytes(bytes),
  m_kind(kind)
{}

const_block::const_block(const void* data, std::size_t bytes, memory_kind_type kind) :
  m_data(const_cast<void*>(data)),
  m_bytes(bytes),
  m_kind(kind)
{}

const void* const_block::data() const
{
    return m_data;
}

std::size_t const_block::bytes() const
{
    return m_bytes;
}

memory_kind_type const_block::kind() const
{
    return m_kind;
}

bool const_block::empty() const
{
    return not bool(*this);
}

const_block::operator bool() const
{
    return (m_data != nullptr) && (m_bytes != 0U) && (m_kind != memory_kind_type::none);
}

void* block::data()
{
    return m_data;
}

}  // namespace srf::memory

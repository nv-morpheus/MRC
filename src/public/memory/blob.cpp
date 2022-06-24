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

#include <srf/memory/blob.hpp>

// Non-main includes
#include <srf/memory/blob_storage.hpp>  // for IBlobStorage
#include <srf/memory/memory_kind.hpp>

#include <utility>  // for move

namespace srf::memory {

blob::blob()  = default;
blob::~blob() = default;

blob::blob(std::shared_ptr<IBlobStorage> view) : m_storage(std::move(view)) {}

void* blob::data()
{
    return (m_storage ? m_storage->data() : nullptr);
}

const void* blob::data() const
{
    return (m_storage ? m_storage->data() : nullptr);
}

std::size_t blob::bytes() const
{
    return (m_storage ? m_storage->bytes() : 0UL);
}

memory_kind_type blob::kind() const
{
    return (m_storage ? m_storage->kind() : memory_kind_type::none);
}

bool blob::empty() const
{
    return not bool(*this);
}

blob::operator bool() const
{
    return m_storage && (m_storage->data() != nullptr) && (m_storage->bytes() != 0U);
}

blob blob::allocate(std::size_t bytes) const
{
    return blob(m_storage->allocate(bytes));
}
}  // namespace srf::memory

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdexcept>

namespace mrc::memory {

struct memory_block
{
    memory_block() = default;
    memory_block(const void* data, std::size_t bytes) : m_data(const_cast<void*>(data)), m_bytes(bytes) {}
    virtual ~memory_block() = default;

    memory_block(const memory_block& other)            = default;
    memory_block& operator=(const memory_block& other) = default;

    memory_block(memory_block&&)            = delete;
    memory_block& operator=(memory_block&&) = delete;

    void* data()
    {
        return m_data;
    }

    const void* data() const
    {
        return m_data;
    }

    std::size_t bytes() const
    {
        return m_bytes;
    }

    bool contains(const void* ptr) const
    {
        const auto* p = static_cast<const std::byte*>(ptr);
        auto* s       = static_cast<std::byte*>(m_data);
        auto* e       = s + m_bytes;
        return (m_data != nullptr && s <= p && p < e);
    }

    std::uintptr_t distance(void* ptr)
    {
        if (!contains(ptr))
        {
            throw std::runtime_error("cannot compute distance - ptr not owned by block");
        }
        auto s = reinterpret_cast<std::uintptr_t>(data());
        auto e = reinterpret_cast<std::uintptr_t>(ptr);
        return e - s;
    }

    void* offset(std::size_t distance)
    {
        if (distance > bytes())
        {
            return nullptr;
        }
        auto* mem = static_cast<std::byte*>(data());
        return mem + distance;
    }

  private:
    void* m_data{nullptr};
    std::size_t m_bytes{0};
};

template <typename CompareT = std::less<>>
struct memory_block_compare_size
{
    using is_transparent = void;  // NOLINT

    constexpr bool operator()(std::size_t size, const memory_block& block, CompareT compare = CompareT()) const
    {
        return compare(size, block.bytes());
    }

    constexpr bool operator()(const memory_block& block, std::size_t size, CompareT compare = CompareT()) const
    {
        return compare(block.bytes(), size);
    }

    constexpr bool operator()(const memory_block& lhs, const memory_block& rhs, CompareT compare = CompareT()) const
    {
        if (compare(lhs.bytes(), rhs.bytes()))
        {
            return true;
        }
        if (lhs.bytes() == rhs.bytes() && compare(lhs.data(), rhs.data()))
        {
            return true;
        }
        return false;
    }
};

template <typename CompareT = std::less<>>
struct memory_block_compare_addr
{
    using is_transparent = void;  // NOLINT

    constexpr bool operator()(void* addr, const memory_block& block, CompareT compare = CompareT()) const
    {
        return compare(reinterpret_cast<std::byte*>(addr), block.data());
    }

    constexpr bool operator()(const memory_block& block, void* addr, CompareT compare = CompareT()) const
    {
        return compare(block.data(), reinterpret_cast<std::byte*>(addr));
    }

    constexpr bool operator()(const memory_block& lhs, const memory_block& rhs, CompareT compare = CompareT()) const
    {
        return compare(lhs.data(), rhs.data());
    }
};

}  // namespace mrc::memory

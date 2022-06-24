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

#include "srf/memory/core/memory_block.hpp"

#include <glog/logging.h>

#include <cstddef>
#include <map>
#include <queue>
#include <utility>

namespace srf::memory {

template <typename BlockType>
class block_manager final
{
    static_assert(std::is_base_of<memory_block, BlockType>::value, "should be derived from memory_block");

  public:
    using block_type = BlockType;

    block_manager()  = default;
    ~block_manager() = default;

    block_manager(block_manager&& other) noexcept : m_block_map(std::move(other.m_block_map)) {}

    block_manager& operator=(block_manager&& other)
    {
        m_block_map = std::move(other.m_block_map);
        return *this;
    }

    block_manager(const block_manager&) = delete;
    block_manager& operator=(const block_manager&) = delete;

    const block_type& add_block(block_type&& block)
    {
        auto key = reinterpret_cast<std::uintptr_t>(block.data()) + block.bytes();
        DCHECK(!owns(block.data()) && !owns(reinterpret_cast<void*>(key - 1)))
            << "block manager already owns a block with an overlapping address";
        DVLOG(1) << "adding block: " << key << " - " << block.data() << "; " << block.bytes();
        m_block_map[key] = std::move(block);
        return m_block_map[key];
    }

    const block_type* find_block(void* ptr) const
    {
        auto search = find_entry(ptr);
        if (search != m_block_map.end() && search->second.contains(ptr))
        {
            DVLOG(3) << this << ": block found";
            return &search->second;
        }
        DVLOG(3) << this << ": no block found for " << ptr;
        return nullptr;
    }

    void drop_block(void* ptr)
    {
        DVLOG(1) << "dropping block: " << ptr;
        auto search = find_entry(ptr);
        if (search != m_block_map.end() && search->second.contains(ptr))
        {
            DVLOG(3) << "found block; dropping block: " << search->first << "; " << search->second.data();
            m_block_map.erase(search);
        }
    }

    auto size() const noexcept
    {
        return m_block_map.size();
    }

    void clear() noexcept
    {
        DVLOG(2) << "clearing block map";
        m_block_map.clear();
    }

    std::vector<BlockType> blocks() const noexcept
    {
        DVLOG(2) << "getting a vector of blocks - " << m_block_map.size();
        std::vector<BlockType> v;
        v.reserve(m_block_map.size());
        for (const auto& it : m_block_map)
        {
            v.push_back(it.second.pointer());
        }
        return v;
    }

    bool owns(void* addr)
    {
        auto block = find_block(addr);
        return (block && block->contains(addr));
    }

  private:
    inline auto find_entry(const void* ptr) const
    {
        DVLOG(3) << "looking for block containing: " << ptr;
        auto key = reinterpret_cast<std::uintptr_t>(ptr);
        return m_block_map.upper_bound(key);
    }

    // todo: used a static block allocator here to avoid allocation issues
    std::map<std::uintptr_t, block_type> m_block_map;
};

}  // namespace srf::memory

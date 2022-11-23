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

#include "./shared_resource_bit_map.hpp"

#include "mrc/core/bitmap.hpp"

#include <cstdint>  // for uint32_t
#include <utility>  // for pair

namespace mrc {

void SharedResourceBitMap::insert(const Bitmap& bitmap, const std::uint32_t& object_id)
{
    bitmap.for_each_bit([this, object_id](std::uint32_t, std::uint32_t bit_index) { m_map[bit_index].on(object_id); });
}

void SharedResourceBitMap::for_objects(std::uint32_t bit_index, std::function<void(const std::uint32_t&)> lambda) const
{
    auto search = m_map.find(bit_index);
    if (search != m_map.end())
    {
        const auto& bitmap = search->second;
        bitmap.for_each_bit([&](std::uint32_t, std::uint32_t bit_index) { lambda(bit_index); });
    }
}

std::size_t SharedResourceBitMap::object_count(std::uint32_t bit_index) const
{
    std::size_t count = 0;
    for_objects(bit_index, [&](std::uint32_t) mutable { ++count; });
    return count;
}

Bitmap SharedResourceBitMap::bitmap(std::uint32_t bit_index) const
{
    auto search = m_map.find(bit_index);
    if (search != m_map.end())
    {
        return search->second;
    }
    return Bitmap();
}

const std::map<std::uint32_t, Bitmap>& SharedResourceBitMap::map() const
{
    return m_map;
}
}  // namespace mrc

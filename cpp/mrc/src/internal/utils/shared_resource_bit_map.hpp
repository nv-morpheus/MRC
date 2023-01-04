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

#include "mrc/core/bitmap.hpp"

#include <cstddef>  // for size_t
#include <cstdint>  // for uint32_t
#include <functional>
#include <map>

namespace mrc {

class SharedResourceBitMap
{
  public:
    void insert(const Bitmap&, const std::uint32_t& object_id);

    [[nodiscard]] std::size_t object_count(std::uint32_t bit_index) const;
    void for_objects(std::uint32_t bit_index, std::function<void(const std::uint32_t&)>) const;

    [[nodiscard]] Bitmap bitmap(std::uint32_t bit_index) const;

    const std::map<std::uint32_t, Bitmap>& map() const;

  private:
    std::map<std::uint32_t, Bitmap> m_map;
};

}  // namespace mrc

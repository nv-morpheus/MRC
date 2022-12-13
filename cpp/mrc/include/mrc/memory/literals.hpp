/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// MODIFICATION MESSAGE

// Modification Notes:
// - added TiB and TB
// - taken from foonathan/memory (memory_arena.hpp)
// Original Source: https://github.com/foonathan/memory
//
// Original License:
/*
Copyright (C) 2015-2020 Jonathan Müller <jonathanmueller.dev@gmail.com>

This software is provided 'as-is', without any express or
implied warranty. In no event will the authors be held
liable for any damages arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute
it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented;
   you must not claim that you wrote the original software.
   If you use this software in a product, an acknowledgment
   in the product documentation would be appreciated but
   is not required.

2. Altered source versions must be plainly marked as such,
   and must not be misrepresented as being the original software.

3. This notice may not be removed or altered from any
   source distribution.
*/

// Copyright (C) 2015-2016 Jonathan Müller <jonathanmueller.dev@gmail.com>
// This file is subject to the license terms in the LICENSE file
// found in the top-level directory of this distribution.

#pragma once

#include <cstdint>

namespace mrc::memory::literals {

constexpr std::size_t operator"" _KiB(unsigned long long value) noexcept
{
    return std::size_t(value * 1024);
}

constexpr std::size_t operator"" _KB(unsigned long long value) noexcept
{
    return std::size_t(value * 1000);
}

constexpr std::size_t operator"" _MiB(unsigned long long value) noexcept
{
    return std::size_t(value * 1024 * 1024);
}

constexpr std::size_t operator"" _MB(unsigned long long value) noexcept
{
    return std::size_t(value * 1000 * 1000);
}

constexpr std::size_t operator"" _GiB(unsigned long long value) noexcept
{
    return std::size_t(value * 1024 * 1024 * 1024);
}

constexpr std::size_t operator"" _GB(unsigned long long value) noexcept
{
    return std::size_t(value * 1000 * 1000 * 1000);
}

constexpr std::size_t operator"" _TiB(unsigned long long value) noexcept
{
    return std::size_t(value * 1024 * 1024 * 1024 * 1024);
}

constexpr std::size_t operator"" _TB(unsigned long long value) noexcept
{
    return std::size_t(value * 1000 * 1000 * 1000 * 1000);
}
}  // namespace mrc::memory::literals

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

#include <cstdlib>

namespace mrc::sfinae {

#define MRC_AUTO_RETURN_TYPE(Expr, T)                                                                      \
    decltype(Expr)                                                                                         \
    {                                                                                                      \
        static_assert(std::is_same<decltype(Expr), T>::value, #Expr " does not have the return type " #T); \
        return Expr;                                                                                       \
    }

struct error  // NOLINT(readability-identifier-naming)
{
    operator void*() const noexcept
    {
        // UNREACHABLE("this is just to hide an error and move static_assert to the front");
        std::abort();
        return nullptr;
    }
};

struct l1_concept : error  // NOLINT(readability-identifier-naming)
{};

struct l2_concept : l1_concept  // NOLINT(readability-identifier-naming)
{};

struct l3_concept : l2_concept  // NOLINT(readability-identifier-naming)
{};

struct l4_concept : l3_concept  // NOLINT(readability-identifier-naming)
{};

struct full_concept : l4_concept  // NOLINT(readability-identifier-naming)
{};

template <typename T>
struct invalid_concept  // NOLINT(readability-identifier-naming)
{
    static const bool error = false;  // NOLINT(readability-identifier-naming)
};

}  // namespace mrc::sfinae

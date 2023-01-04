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

#include "mrc/runnable/forward.hpp"

#include <cstdlib>
#include <type_traits>
#include <utility>

namespace mrc::runnable::detail {

template <typename T>
struct self  // NOLINT
{
    using type = T;  // NOLINT
};

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

template <typename ContextT>
auto get_runnable_context_t(RunnableWithContext<ContextT>& not_used)
{
    return self<ContextT>{};
}

struct ctx_fiber  // NOLINT
{};
struct ctx_thread  // NOLINT
{};

template <typename T>
static auto unwrap_context(full_concept c, FiberContext<T>& t)
{
    return std::make_pair(self<ctx_fiber>{}, self<T>{});
}

template <typename T>
static auto unwrap_context(l4_concept c, ThreadContext<T>& t)
{
    return std::make_pair(self<ctx_thread>{}, self<T>{});
}

static auto unwrap_context(l2_concept c, Context& t)
{
    return std::make_pair(self<Context>{}, self<Context>{});
}

template <typename T>
static error unwrap_context(error e, T& t)
{
    static_assert(invalid_concept<T>::error, "object is not a Context");
    return {};
}

template <typename T>
using unwrap_context_first_t =
    typename decltype(detail::unwrap_context(std::declval<full_concept&>(), std::declval<T&>()).first)::type;

template <typename T>
using unwrap_context_second_t =
    typename decltype(detail::unwrap_context(std::declval<full_concept&>(), std::declval<T&>()).second)::type;

template <typename T, typename = void>
struct is_fiber_context : std::false_type
{};

template <typename T>
struct is_fiber_context<T, typename std::enable_if_t<std::is_same_v<ctx_fiber, unwrap_context_first_t<T>>>>
  : std::true_type
{};

template <typename T, typename = void>
struct is_thread_context : std::false_type
{};

template <typename T>
struct is_thread_context<T, typename std::enable_if_t<std::is_same_v<ctx_thread, unwrap_context_first_t<T>>>>
  : std::true_type
{};

template <typename T, typename = void>
struct is_unwrapped_context : std::false_type
{};

template <typename T>
struct is_unwrapped_context<
    T,
    typename std::enable_if_t<std::is_same_v<unwrap_context_first_t<T>, unwrap_context_second_t<T>>>> : std::true_type
{};

}  // namespace mrc::runnable::detail

/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * Original Source: https://github.com/jbaldwin/libcoro
 * Original License: Apache License, Version 2.0; included below
 */

/**
 * Copyright 2021 Josh Baldwin
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <concepts>
#include <coroutine>
#include <type_traits>
#include <utility>

namespace mrc::coroutines::concepts {
/**
 * This concept declares a type that is required to meet the c++20 coroutine operator co_await()
 * retun type.  It requires the following three member functions:
 *      await_ready() -> bool
 *      await_suspend(std::coroutine_handle<>) -> void|bool|std::coroutine_handle<>
 *      await_resume() -> decltype(auto)
 *          Where the return type on await_resume is the requested return of the awaitable.
 */
// clang-format off
template<typename T>
concept awaiter = requires(T t, std::coroutine_handle<> c)
{
    { t.await_ready() } -> std::same_as<bool>;
    requires std::same_as<decltype(t.await_suspend(c)), void> ||
             std::same_as<decltype(t.await_suspend(c)), bool> ||
             std::same_as<decltype(t.await_suspend(c)), std::coroutine_handle<>>;
    { t.await_resume() };
};

/**
 * This concept declares a type that can be operator co_await()'ed and returns an awaiter_type.
 */
template<typename T>
concept awaitable = requires(T t)
{
    // operator co_await()
    { t.operator co_await() } -> awaiter;
};

template<typename T>
concept awaiter_void = requires(T t, std::coroutine_handle<> c)
{
    { t.await_ready() } -> std::same_as<bool>;
    requires std::same_as<decltype(t.await_suspend(c)), void> ||
        std::same_as<decltype(t.await_suspend(c)), bool> ||
        std::same_as<decltype(t.await_suspend(c)), std::coroutine_handle<>>;
    {t.await_resume()} -> std::same_as<void>;
};

template<typename T>
concept awaitable_void = requires(T t)
{
    // operator co_await()
    { t.operator co_await() } -> awaiter_void;
};

template<awaitable AwaitableT, typename = void>
struct awaitable_traits
{
};

template<awaitable AwaitableT>
static auto get_awaiter(AwaitableT&& value)
{
    return std::forward<AwaitableT>(value).operator co_await();
}

template<awaitable AwaitableT>
struct awaitable_traits<AwaitableT>
{
    using awaiter_type        = decltype(get_awaiter(std::declval<AwaitableT>()));
    using awaiter_return_type = decltype(std::declval<awaiter_type>().await_resume());
};

template<typename T, typename U>
concept awaitable_return_same_as = requires(T t)
{
    requires awaitable<T>;
    requires std::same_as<std::remove_reference_t<typename awaitable_traits<T>::awaiter_return_type>, U>;
};
// clang-format on

}  // namespace mrc::coroutines::concepts

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

#include "mrc/forward.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/detail/type_traits.hpp"
#include "mrc/utils/macros.hpp"

namespace mrc::runnable {

template <typename T>
using runnable_context_t = typename decltype(detail::get_runnable_context_t(std::declval<T&>()))::type;

template <typename T>
using is_unwrapped_context = detail::is_unwrapped_context<T>;  // NOLINT

template <typename T>
inline constexpr bool is_unwrapped_context_v = is_unwrapped_context<T>::value;  // NOLINT

template <typename T>
using is_fiber_context = detail::is_fiber_context<T>;  // NOLINT

template <typename T>
inline constexpr bool is_fiber_context_v = is_fiber_context<T>::value;  // NOLINT

template <typename T>
using is_thread_context = detail::is_thread_context<T>;  // NOLINT

template <typename T>
inline constexpr bool is_thread_context_v = is_thread_context<T>::value;  // NOLINT

template <typename T>
using is_fiber_runnable = is_fiber_context<runnable_context_t<T>>;  // NOLINT

template <typename T>
inline constexpr bool is_fiber_runnable_v = is_fiber_runnable<T>::value;  // NOLINT

template <typename T>
using is_thread_runnable = is_thread_context<runnable_context_t<T>>;  // NOLINT

template <typename T>
inline constexpr bool is_thread_runnable_v = is_thread_runnable<T>::value;  // NOLINT

template <typename T>
using unwrap_context_t = typename detail::unwrap_context_second_t<T>;

}  // namespace mrc::runnable

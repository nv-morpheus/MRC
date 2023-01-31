/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/coroutines/concepts/awaitable.hpp"

#include <concepts>

namespace mrc::coroutines::concepts {

// clang-format off
template<typename T, typename ReturnT>
concept promise = requires(T t)
{
    { t.get_return_object() } -> std::convertible_to<std::coroutine_handle<>>;
    { t.initial_suspend() } -> awaiter;
    { t.final_suspend() } -> awaiter;
    { t.yield_value() } -> awaitable;
}
&& requires(T t, ReturnT return_value)
{
    requires std::same_as<decltype(t.return_void()), void> ||
             std::same_as<decltype(t.return_value(return_value)), void> ||
             requires { t.yield_value(return_value); };
};
// clang-format on

}  // namespace mrc::coroutines::concepts

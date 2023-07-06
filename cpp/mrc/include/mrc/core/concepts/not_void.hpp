/*
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

#pragma once

#include "mrc/type_traits.hpp"

#include <concepts>

namespace mrc::core::concepts {

template <typename T>
concept not_void = requires { requires not std::same_as<T, void>; };

template <typename T>
concept is_shared_ptr = mrc::is_shared_ptr_v<T>;

template <typename T>
concept is_unique_ptr = mrc::is_unique_ptr_v<T>;

template <typename T>
concept is_smart_ptr = mrc::is_shared_ptr_v<T> || mrc::is_unique_ptr_v<T>;

}  // namespace mrc::core::concepts

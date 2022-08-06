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

#include <concepts>
#include <memory>
#include <vector>

#if __cplusplus >= 202002L

template <typename T>
concept vector_like = requires(T t)
{
    t.begin();
    t.reserve(1);
    t.data();
};

static_assert(vector_like<std::vector<int>>);
static_assert(!vector_like<int>);

template <typename T>
concept smart_ptr_like = requires(T t)
{
    t.operator*();
    t.operator->();
    t.release();
    t.reset();
    typename T::element_type;

    requires !std::copyable<T>;

    // clang-format off
    { t.operator->() } -> std::same_as<typename std::add_pointer<typename T::element_type>::type>;
    { *t } -> std::same_as<typename std::add_lvalue_reference<typename T::element_type>::type>;
    // clang-format on
};

static_assert(smart_ptr_like<std::unique_ptr<int>>);
static_assert(!smart_ptr_like<std::shared_ptr<int>>);
static_assert(!smart_ptr_like<int>);

#endif

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

#include "mrc/channel/status.hpp"

#include <type_traits>  // IWYU pragma: export
#include <utility>

namespace mrc::channel {

/**
 * @brief Opaque handle that is a type-erased common ancestor for all Ingress<T>
 */
struct IngressHandle
{
    virtual ~IngressHandle() = default;
};

/**
 * @brief Interface for data flowing into a channel via a fiber yielding write mechanism
 *
 * @tparam T
 */
template <typename T>
struct Ingress : public IngressHandle
{
    ~Ingress() override = default;

    virtual Status await_write(T&&) = 0;

    // If the above overload cannot be matched, copy by value and move into the await_write(T&&) overload. This is only
    // necessary for lvalues. The template parameters give it lower priority in overload resolution.
    template <typename TT = T, typename = std::enable_if_t<std::is_copy_constructible_v<TT>>>
    inline Status await_write(T t)
    {
        return await_write(std::move(t));
    }
};

}  // namespace mrc::channel

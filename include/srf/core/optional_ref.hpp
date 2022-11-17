/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "srf/concepts/invocable.hpp"
#include "srf/concepts/referenceable.hpp"
#include "srf/concepts/types.hpp"
#include "srf/core/error.hpp"
#include "srf/core/std23_expected.hpp"

#include <glog/logging.h>
#include <spdlog/fmt/bundled/format.h>

#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <type_traits>

namespace srf {

template <concepts::referenceable T>
class OptionalRef final
{
  public:
    constexpr OptionalRef() noexcept : m_pointer(nullptr) {}
    constexpr OptionalRef(std::nullopt_t no) noexcept : m_pointer(nullptr) {}
    explicit constexpr OptionalRef(T& t) noexcept : m_pointer(std::addressof(t)) {}
    explicit constexpr OptionalRef(std::in_place_t ip, T& t) noexcept : m_pointer(std::addressof(t)) {}

    ~OptionalRef() = default;

    constexpr OptionalRef(const OptionalRef& other) noexcept : m_pointer(other.m_pointer) {}
    constexpr OptionalRef(OptionalRef&& other) noexcept = delete;

    OptionalRef& operator=(const OptionalRef& other)
    {
        m_pointer = other.m_pointer;
        return *this;
    }

    OptionalRef& operator=(OptionalRef&& other) noexcept
    {
        m_pointer = std::exchange(other.m_pointer, nullptr);
        return *this;
    }

    explicit constexpr operator bool() const noexcept
    {
        return m_pointer != nullptr;
    }

    constexpr T& operator*() const requires(!std::is_class_v<T>)
    {
        if (m_pointer == nullptr)
        {
            throw Error::create(ErrorCode::NullOptional);
        }
        return *m_pointer;
    }

    constexpr T* operator->() const requires std::is_class_v<T>
    {
        if (m_pointer == nullptr)
        {
            throw Error::create(ErrorCode::NullOptional);
        }
        return m_pointer;
    }

    void reset() noexcept
    {
        m_pointer = nullptr;
    }

    void reset(T& t)
    {
        m_pointer = std::addressof(t);
    }

    template <concepts::invocable<T&> F, class V = T&, class U = std::remove_cvref_t<std::invoke_result_t<F, V>>>
    constexpr auto and_then(F&& fn) & -> Expected<U>
    {
        if (m_pointer == nullptr)
        {
            return Error::create(ErrorCode::NullOptional);
        }

        if constexpr (std::is_same_v<U, void>)
        {
            std::invoke(std::forward<decltype(fn)>(fn), *m_pointer);
            return {};
        }
        else
        {
            return std::invoke(std::forward<decltype(fn)>(fn), *m_pointer);
        }
    }

    template <concepts::invocable<T const&> F,
              class V = T const&,
              class U = std::remove_cvref_t<std::invoke_result_t<F, V>>>
    constexpr auto and_then(F&& fn) const& -> Expected<U>
    {
        if (m_pointer == nullptr)
        {
            return Error::create(ErrorCode::NullOptional);
        }

        if constexpr (std::is_same_v<U, void>)
        {
            std::invoke(std::forward<decltype(fn)>(fn), *m_pointer);
            return {};
        }
        else
        {
            return std::invoke(std::forward<decltype(fn)>(fn), *m_pointer); 
        }
    }

  private:
    T* m_pointer;
};

}  // namespace srf

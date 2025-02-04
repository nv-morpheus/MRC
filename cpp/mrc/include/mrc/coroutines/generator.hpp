/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <coroutine>
#include <memory>
#include <type_traits>

namespace mrc::coroutines {

template <typename T>
class Generator;

namespace detail {

template <typename T>
class GeneratorPromise
{
  public:
    using value_type     = std::remove_reference_t<T>;
    using reference_type = std::conditional_t<std::is_reference_v<T>, T, T&>;
    using pointer_type   = value_type*;

    GeneratorPromise() = default;

    auto get_return_object() noexcept -> Generator<T>;

    auto initial_suspend() const
    {
        return std::suspend_always{};
    }

    auto final_suspend() const noexcept(true)
    {
        return std::suspend_always{};
    }

    template <typename U = T, std::enable_if_t<!std::is_rvalue_reference<U>::value, int> = 0>
    auto yield_value(std::remove_reference_t<T>& value) noexcept
    {
        m_value = std::addressof(value);
        return std::suspend_always{};
    }

    auto yield_value(std::remove_reference_t<T>&& value) noexcept
    {
        m_value = std::addressof(value);
        return std::suspend_always{};
    }

    auto unhandled_exception() -> void
    {
        m_exception = std::current_exception();
    }

    auto return_void() noexcept -> void {}

    auto value() const noexcept -> reference_type
    {
        return static_cast<reference_type>(*m_value);
    }

    template <typename U>
    auto await_transform(U&& value) -> std::suspend_never = delete;

    auto rethrow_if_exception() -> void
    {
        if (m_exception)
        {
            std::rethrow_exception(m_exception);
        }
    }

  private:
    pointer_type m_value{nullptr};
    std::exception_ptr m_exception;
};

struct GeneratorSentinel
{};

template <typename T>
class GeneratorIterator
{
    using coroutine_handle = std::coroutine_handle<GeneratorPromise<T>>;  // NOLINT

  public:
    using iterator_category = std::input_iterator_tag;  // NOLINT
    using difference_type   = std::ptrdiff_t;
    using value_type        = typename GeneratorPromise<T>::value_type;
    using reference         = typename GeneratorPromise<T>::reference_type;  // NOLINT
    using pointer           = typename GeneratorPromise<T>::pointer_type;    // NOLINT

    GeneratorIterator() noexcept = default;

    explicit GeneratorIterator(coroutine_handle coroutine) noexcept : m_coroutine(coroutine) {}

    friend auto operator==(const GeneratorIterator& it, GeneratorSentinel /*unused*/) noexcept -> bool
    {
        return it.m_coroutine == nullptr || it.m_coroutine.done();
    }

    friend auto operator!=(const GeneratorIterator& it, GeneratorSentinel s) noexcept -> bool
    {
        return !(it == s);
    }

    friend auto operator==(GeneratorSentinel s, const GeneratorIterator& it) noexcept -> bool
    {
        return (it == s);
    }

    friend auto operator!=(GeneratorSentinel s, const GeneratorIterator& it) noexcept -> bool
    {
        return it != s;
    }

    GeneratorIterator& operator++()
    {
        m_coroutine.resume();
        if (m_coroutine.done())
        {
            m_coroutine.promise().rethrow_if_exception();
        }

        return *this;
    }

    auto operator++(int) -> void
    {
        (void)operator++();
    }

    reference operator*() const noexcept
    {
        return m_coroutine.promise().value();
    }

    pointer operator->() const noexcept
    {
        return std::addressof(operator*());
    }

  private:
    coroutine_handle m_coroutine{nullptr};
};

}  // namespace detail

template <typename T>
class Generator
{
  public:
    using promise_type = detail::GeneratorPromise<T>;
    using iterator     = detail::GeneratorIterator<T>;  // NOLINT
    using sentinel     = detail::GeneratorSentinel;     // NOLINT

    Generator() noexcept : m_coroutine(nullptr) {}

    Generator(const Generator&) = delete;
    Generator(Generator&& other) noexcept : m_coroutine(other.m_coroutine)
    {
        other.m_coroutine = nullptr;
    }

    auto operator=(const Generator&) = delete;
    auto operator=(Generator&& other) noexcept -> Generator&
    {
        m_coroutine       = other.m_coroutine;
        other.m_coroutine = nullptr;

        return *this;
    }

    ~Generator()
    {
        if (m_coroutine)
        {
            m_coroutine.destroy();
        }
    }

    auto begin() -> iterator
    {
        if (m_coroutine != nullptr)
        {
            m_coroutine.resume();
            if (m_coroutine.done())
            {
                m_coroutine.promise().rethrow_if_exception();
            }
        }

        return iterator{m_coroutine};
    }

    auto end() noexcept -> sentinel
    {
        return sentinel{};
    }

  private:
    friend class detail::GeneratorPromise<T>;

    explicit Generator(std::coroutine_handle<promise_type> coroutine) noexcept : m_coroutine(coroutine) {}

    std::coroutine_handle<promise_type> m_coroutine;
};

namespace detail {
template <typename T>
auto GeneratorPromise<T>::get_return_object() noexcept -> Generator<T>
{
    return Generator<T>{std::coroutine_handle<GeneratorPromise<T>>::from_promise(*this)};
}

}  // namespace detail

}  // namespace mrc::coroutines

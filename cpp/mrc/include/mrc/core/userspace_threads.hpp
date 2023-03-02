/**
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <boost/fiber/all.hpp>
#include <boost/fiber/condition_variable.hpp>
#include <boost/fiber/recursive_mutex.hpp>

// Suppress naming conventions in this file to allow matching std and boost libraries
// NOLINTBEGIN(readability-identifier-naming)

namespace mrc::userspace_threads {
namespace detail {

// Base implementation of shared_promise. Follows boost::fibers::promise<T>
template <typename T>
struct shared_promise_base
{
  public:
    shared_promise_base() : m_promise(), m_shared_future(m_promise.get_future()) {}

    void set_exception(std::exception_ptr p)
    {
        m_promise.set_exception(p);
    }

    boost::fibers::shared_future<T> get_future()
    {
        return m_shared_future;
    }

    void swap(shared_promise_base& other) noexcept
    {
        std::swap(m_promise, other.m_promise);
        std::swap(m_shared_future, other.m_shared_future);
    }

  protected:
    boost::fibers::promise<T> m_promise;
    boost::fibers::shared_future<T> m_shared_future;
};

}  // namespace detail

// Wrapper for a promise that holds an internal reference to a shared future. Reduces the need to have both a
// promise and shared future member variables
template <typename T>
struct shared_promise : private detail::shared_promise_base<T>
{
  private:
    using base_t = detail::shared_promise_base<T>;

  public:
    shared_promise() = default;

    void set_value(const T& value)
    {
        base_t::m_promise.set_value(value);
    }

    void set_value(T&& value)
    {
        base_t::m_promise.set_value(std::move(value));
    }

    void swap(shared_promise& other) noexcept
    {
        base_t::swap(other);
    }

    using base_t::get_future;
    using base_t::set_exception;
};

template <typename T>
struct shared_promise<T&> : private detail::shared_promise_base<T&>
{
  private:
    using base_t = detail::shared_promise_base<T&>;

  public:
    shared_promise() = default;

    void set_value(T& value)
    {
        base_t::m_promise.set_value(value);
    }

    void swap(shared_promise& other) noexcept
    {
        base_t::swap(other);
    }

    using base_t::get_future;
    using base_t::set_exception;
};

template <>
struct shared_promise<void> : private detail::shared_promise_base<void>
{
  private:
    using base_t = detail::shared_promise_base<void>;

  public:
    shared_promise() = default;

    void set_value()
    {
        base_t::m_promise.set_value();
    }

    void swap(shared_promise& other) noexcept
    {
        base_t::swap(other);
    }

    using base_t::get_future;
    using base_t::set_exception;
};

using mutex = boost::fibers::mutex;

using recursive_mutex = boost::fibers::recursive_mutex;

using cv = boost::fibers::condition_variable;

using cv_any = boost::fibers::condition_variable_any;

using launch = boost::fibers::launch;

template <typename T>
using promise = boost::fibers::promise<T>;

// template <typename T>
// using shared_promise = userspace::shared_promise<T>;

template <typename T>
using future = boost::fibers::future<T>;

template <typename T>
using shared_future = boost::fibers::shared_future<T>;

template <class R, class... Args>
using packaged_task = boost::fibers::packaged_task<R(Args...)>;

template <class Function, class... Args>
static auto async(Function&& f, Args&&... args)
{
    return boost::fibers::async(f, std::forward<Args>(args)...);
}

template <typename Rep, typename Period>
static void sleep_for(std::chrono::duration<Rep, Period> const& timeout_duration)
{
    boost::this_fiber::sleep_for(timeout_duration);
}

template <typename Clock, typename Duration>
static void sleep_until(std::chrono::time_point<Clock, Duration> const& sleep_time_point)
{
    boost::this_fiber::sleep_until(sleep_time_point);
}

}  // namespace mrc::userspace_threads

// NOLINTEND(readability-identifier-naming)

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

#include "mrc/coroutines/concepts/awaitable.hpp"
#include "mrc/coroutines/when_all.hpp"

#include <condition_variable>
#include <mutex>
#include <type_traits>
#include <utility>

namespace mrc::coroutines {

namespace detail {

class SyncWaitEvent
{
  public:
    SyncWaitEvent(bool initially_set = false);
    SyncWaitEvent(const SyncWaitEvent&)                    = delete;
    SyncWaitEvent(SyncWaitEvent&&)                         = delete;
    auto operator=(const SyncWaitEvent&) -> SyncWaitEvent& = delete;
    auto operator=(SyncWaitEvent&&) -> SyncWaitEvent&      = delete;
    ~SyncWaitEvent()                                       = default;

    auto set() noexcept -> void;
    auto reset() noexcept -> void;
    auto wait() noexcept -> void;

  private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_set{false};
};

class SyncWaitTaskPromiseBase
{
  public:
    SyncWaitTaskPromiseBase() noexcept = default;
    virtual ~SyncWaitTaskPromiseBase() = default;

    constexpr static auto initial_suspend() noexcept -> std::suspend_always
    {
        return {};
    }

    auto unhandled_exception() -> void
    {
        m_exception = std::current_exception();
    }

  protected:
    SyncWaitEvent* m_event{nullptr};
    std::exception_ptr m_exception;
};

template <typename ReturnT>
class SyncWaitTaskPromise : public SyncWaitTaskPromiseBase
{
  public:
    using coroutine_type = std::coroutine_handle<SyncWaitTaskPromise<ReturnT>>;

    SyncWaitTaskPromise() noexcept  = default;
    ~SyncWaitTaskPromise() override = default;

    auto start(SyncWaitEvent& event)
    {
        m_event = &event;
        coroutine_type::from_promise(*this).resume();
    }

    auto get_return_object() noexcept
    {
        return coroutine_type::from_promise(*this);
    }

    auto yield_value(ReturnT&& value) noexcept
    {
        m_return_value = std::addressof(value);
        return final_suspend();
    }

    auto final_suspend() noexcept
    {
        struct CompletionNotifier
        {
            auto await_ready() const noexcept
            {
                return false;
            }
            auto await_suspend(coroutine_type coroutine) const noexcept
            {
                coroutine.promise().m_event->set();
            }
            auto await_resume() noexcept {};
        };

        return CompletionNotifier{};
    }

    auto result() -> ReturnT&&
    {
        if (m_exception)
        {
            std::rethrow_exception(m_exception);
        }

        return static_cast<ReturnT&&>(*m_return_value);
    }

  private:
    std::remove_reference_t<ReturnT>* m_return_value;
};

template <>
class SyncWaitTaskPromise<void> : public SyncWaitTaskPromiseBase
{
    using coroutine_type = std::coroutine_handle<SyncWaitTaskPromise<void>>;

  public:
    SyncWaitTaskPromise() noexcept  = default;
    ~SyncWaitTaskPromise() override = default;

    auto start(SyncWaitEvent& event)
    {
        m_event = &event;
        coroutine_type::from_promise(*this).resume();
    }

    auto get_return_object() noexcept
    {
        return coroutine_type::from_promise(*this);
    }

    static auto final_suspend() noexcept
    {
        struct CompletionNotifier
        {
            constexpr static auto await_ready() noexcept
            {
                return false;
            }

            static auto await_suspend(coroutine_type coroutine) noexcept
            {
                coroutine.promise().m_event->set();
            }

            constexpr static auto await_resume() noexcept {};
        };

        return CompletionNotifier{};
    }

    auto return_void() noexcept -> void {}

    auto result() -> void
    {
        if (m_exception)
        {
            std::rethrow_exception(m_exception);
        }
    }
};

template <typename ReturnT>
class SyncWaitTask
{
  public:
    using promise_type   = SyncWaitTaskPromise<ReturnT>;
    using coroutine_type = std::coroutine_handle<promise_type>;

    SyncWaitTask(coroutine_type coroutine) noexcept : m_coroutine(coroutine) {}

    SyncWaitTask(const SyncWaitTask&) = delete;
    SyncWaitTask(SyncWaitTask&& other) noexcept : m_coroutine(std::exchange(other.m_coroutine, coroutine_type{})) {}
    auto operator=(const SyncWaitTask&) -> SyncWaitTask& = delete;
    auto operator=(SyncWaitTask&& other) -> SyncWaitTask&
    {
        if (std::addressof(other) != this)
        {
            m_coroutine = std::exchange(other.m_coroutine, coroutine_type{});
        }

        return *this;
    }

    ~SyncWaitTask()
    {
        if (m_coroutine)
        {
            m_coroutine.destroy();
        }
    }

    auto start(SyncWaitEvent& event) noexcept
    {
        m_coroutine.promise().start(event);
    }

    auto return_value() -> decltype(auto)
    {
        if constexpr (std::is_same_v<void, ReturnT>)
        {
            // Propagate exceptions.
            m_coroutine.promise().result();
            return;
        }
        else
        {
            return std::remove_reference_t<ReturnT>{std::move(m_coroutine.promise().result())};
        }
    }

  private:
    coroutine_type m_coroutine;
};

template <concepts::awaitable AwaitableT,
          typename ReturnT = typename concepts::awaitable_traits<AwaitableT>::awaiter_return_type>
static auto make_sync_wait_task(AwaitableT&& a) -> SyncWaitTask<ReturnT>;

template <concepts::awaitable AwaitableT, typename ReturnT>
static auto make_sync_wait_task(AwaitableT&& a) -> SyncWaitTask<ReturnT>
{
    if constexpr (std::is_void_v<ReturnT>)
    {
        co_await std::forward<AwaitableT>(a);
        co_return;
    }
    else
    {
        co_yield co_await std::forward<AwaitableT>(a);
    }
}

}  // namespace detail

template <concepts::awaitable AwaitableT>
auto sync_wait(AwaitableT&& a) -> decltype(auto)
{
    detail::SyncWaitEvent e{};
    // we force the lvalue ref path on the co_await operator of the AwaitableT
    // on this path, the return value is maintained as part of the AwaitableT
    // only after the AwaitableT is complete do we transfer the return value
    // from the AwaitableT back to the user
    AwaitableT ca = std::forward<AwaitableT>(a);
    auto task     = detail::make_sync_wait_task(ca);
    task.start(e);
    e.wait();

    return task.return_value();
}

}  // namespace mrc::coroutines

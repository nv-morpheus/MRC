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
#include "mrc/coroutines/detail/void_value.hpp"

#include <atomic>
#include <coroutine>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <vector>

namespace mrc::coroutines {

namespace detail {

class WhenAllLatch
{
  public:
    WhenAllLatch(std::size_t count) noexcept : m_count(count + 1) {}

    WhenAllLatch(const WhenAllLatch&) = delete;
    WhenAllLatch(WhenAllLatch&& other) :
      m_count(other.m_count.load(std::memory_order::acquire)),
      m_awaiting_coroutine(std::exchange(other.m_awaiting_coroutine, nullptr))
    {}

    auto operator=(const WhenAllLatch&) -> WhenAllLatch& = delete;
    auto operator=(WhenAllLatch&& other) -> WhenAllLatch&
    {
        if (std::addressof(other) != this)
        {
            m_count.store(other.m_count.load(std::memory_order::acquire), std::memory_order::relaxed);
            m_awaiting_coroutine = std::exchange(other.m_awaiting_coroutine, nullptr);
        }

        return *this;
    }

    auto is_ready() const noexcept -> bool
    {
        return m_awaiting_coroutine != nullptr && m_awaiting_coroutine.done();
    }

    auto try_await(std::coroutine_handle<> awaiting_coroutine) noexcept -> bool
    {
        m_awaiting_coroutine = awaiting_coroutine;
        return m_count.fetch_sub(1, std::memory_order::acq_rel) > 1;
    }

    auto notify_awaitable_completed() noexcept -> void
    {
        if (m_count.fetch_sub(1, std::memory_order::acq_rel) == 1)
        {
            m_awaiting_coroutine.resume();
        }
    }

  private:
    /// The number of tasks that are being waited on.
    std::atomic<std::size_t> m_count;
    /// The when_all_task awaiting to be resumed upon all task completions.
    std::coroutine_handle<> m_awaiting_coroutine{nullptr};
};

template <typename TaskContainerT>
class WhenAllReadyAwaitable;

template <typename ReturnT>
class when_all_task;

/// Empty tuple<> implementation.
template <>
class WhenAllReadyAwaitable<std::tuple<>>
{
  public:
    constexpr WhenAllReadyAwaitable() noexcept = default;
    explicit constexpr WhenAllReadyAwaitable(std::tuple<> /*unused*/) noexcept {}

    static constexpr auto await_ready() noexcept -> bool
    {
        return true;
    }

    static constexpr auto await_suspend(std::coroutine_handle<> /*unused*/) noexcept -> void {}

    static constexpr auto await_resume() noexcept -> std::tuple<>
    {
        return {};
    }
};

template <typename... TaskTypesT>
class WhenAllReadyAwaitable<std::tuple<TaskTypesT...>>
{
  public:
    explicit WhenAllReadyAwaitable(TaskTypesT&&... tasks) noexcept(
        std::conjunction<std::is_nothrow_move_constructible<TaskTypesT>...>::value) :
      m_latch(sizeof...(TaskTypesT)),
      m_tasks(std::move(tasks)...)
    {}

    explicit WhenAllReadyAwaitable(std::tuple<TaskTypesT...>&& tasks) noexcept(
        std::is_nothrow_move_constructible_v<std::tuple<TaskTypesT...>>) :
      m_latch(sizeof...(TaskTypesT)),
      m_tasks(std::move(tasks))
    {}

    WhenAllReadyAwaitable(const WhenAllReadyAwaitable&) = delete;
    WhenAllReadyAwaitable(WhenAllReadyAwaitable&& other) :
      m_latch(std::move(other.m_latch)),
      m_tasks(std::move(other.m_tasks))
    {}

    auto operator=(const WhenAllReadyAwaitable&) -> WhenAllReadyAwaitable& = delete;
    auto operator=(WhenAllReadyAwaitable&&) -> WhenAllReadyAwaitable&      = delete;

    auto operator co_await() & noexcept
    {
        struct Awaiter
        {
            explicit Awaiter(WhenAllReadyAwaitable& awaitable) noexcept : m_awaitable(awaitable) {}

            auto await_ready() const noexcept -> bool
            {
                return m_awaitable.is_ready();
            }

            auto await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> bool
            {
                return m_awaitable.try_await(awaiting_coroutine);
            }

            auto await_resume() noexcept -> std::tuple<TaskTypesT...>&
            {
                return m_awaitable.m_tasks;
            }

          private:
            WhenAllReadyAwaitable& m_awaitable;
        };

        return Awaiter{*this};
    }

    auto operator co_await() && noexcept
    {
        struct Awaiter
        {
            explicit Awaiter(WhenAllReadyAwaitable& awaitable) noexcept : m_awaitable(awaitable) {}

            auto await_ready() const noexcept -> bool
            {
                return m_awaitable.is_ready();
            }

            auto await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> bool
            {
                return m_awaitable.try_await(awaiting_coroutine);
            }

            auto await_resume() noexcept -> std::tuple<TaskTypesT...>&&
            {
                return std::move(m_awaitable.m_tasks);
            }

          private:
            WhenAllReadyAwaitable& m_awaitable;
        };

        return Awaiter{*this};
    }

  private:
    auto is_ready() const noexcept -> bool
    {
        return m_latch.is_ready();
    }

    auto try_await(std::coroutine_handle<> awaiting_coroutine) noexcept -> bool
    {
        std::apply(
            [this](auto&&... tasks) {
                ((tasks.start(m_latch)), ...);
            },
            m_tasks);
        return m_latch.try_await(awaiting_coroutine);
    }

    WhenAllLatch m_latch;
    std::tuple<TaskTypesT...> m_tasks;
};

template <typename TaskContainerT>
class WhenAllReadyAwaitable
{
  public:
    explicit WhenAllReadyAwaitable(TaskContainerT&& tasks) noexcept :
      m_latch(std::size(tasks)),
      m_tasks(std::forward<TaskContainerT>(tasks))
    {}

    WhenAllReadyAwaitable(const WhenAllReadyAwaitable&) = delete;
    WhenAllReadyAwaitable(WhenAllReadyAwaitable&& other) noexcept(std::is_nothrow_move_constructible_v<TaskContainerT>) :
      m_latch(std::move(other.m_latch)),
      m_tasks(std::move(m_tasks))
    {}

    auto operator=(const WhenAllReadyAwaitable&) -> WhenAllReadyAwaitable& = delete;
    auto operator=(WhenAllReadyAwaitable&) -> WhenAllReadyAwaitable&       = delete;

    auto operator co_await() & noexcept
    {
        struct Awaiter
        {
            Awaiter(WhenAllReadyAwaitable& awaitable) : m_awaitable(awaitable) {}

            auto await_ready() const noexcept -> bool
            {
                return m_awaitable.is_ready();
            }

            auto await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> bool
            {
                return m_awaitable.try_await(awaiting_coroutine);
            }

            auto await_resume() noexcept -> TaskContainerT&
            {
                return m_awaitable.m_tasks;
            }

          private:
            WhenAllReadyAwaitable& m_awaitable;
        };

        return Awaiter{*this};
    }

    auto operator co_await() && noexcept
    {
        struct Awaiter
        {
            Awaiter(WhenAllReadyAwaitable& awaitable) : m_awaitable(awaitable) {}

            auto await_ready() const noexcept -> bool
            {
                return m_awaitable.is_ready();
            }

            auto await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> bool
            {
                return m_awaitable.try_await(awaiting_coroutine);
            }

            auto await_resume() noexcept -> TaskContainerT&&
            {
                return std::move(m_awaitable.m_tasks);
            }

          private:
            WhenAllReadyAwaitable& m_awaitable;
        };

        return Awaiter{*this};
    }

  private:
    auto is_ready() const noexcept -> bool
    {
        return m_latch.is_ready();
    }

    auto try_await(std::coroutine_handle<> awaiting_coroutine) noexcept -> bool
    {
        for (auto& task : m_tasks)
        {
            task.start(m_latch);
        }

        return m_latch.try_await(awaiting_coroutine);
    }

    WhenAllLatch m_latch;
    TaskContainerT m_tasks;
};

template <typename ReturnT>
class WhenAllTaskPromise
{
  public:
    using coroutine_handle_type = std::coroutine_handle<WhenAllTaskPromise<ReturnT>>;

    WhenAllTaskPromise() noexcept = default;

    auto get_return_object() noexcept
    {
        return coroutine_handle_type::from_promise(*this);
    }

    auto initial_suspend() noexcept -> std::suspend_always
    {
        return {};
    }

    auto final_suspend() noexcept
    {
        struct CompletionNotifier
        {
            auto await_ready() const noexcept -> bool
            {
                return false;
            }
            auto await_suspend(coroutine_handle_type coroutine) const noexcept -> void
            {
                coroutine.promise().m_latch->notify_awaitable_completed();
            }
            auto await_resume() const noexcept {}
        };

        return CompletionNotifier{};
    }

    auto unhandled_exception() noexcept
    {
        m_exception_ptr = std::current_exception();
    }

    auto yield_value(ReturnT&& value) noexcept
    {
        m_return_value = std::addressof(value);
        return final_suspend();
    }

    auto start(WhenAllLatch& latch) noexcept -> void
    {
        m_latch = &latch;
        coroutine_handle_type::from_promise(*this).resume();
    }

    auto return_value() & -> ReturnT&
    {
        if (m_exception_ptr)
        {
            std::rethrow_exception(m_exception_ptr);
        }
        return *m_return_value;
    }

    auto return_value() && -> ReturnT&&
    {
        if (m_exception_ptr)
        {
            std::rethrow_exception(m_exception_ptr);
        }
        return std::forward(*m_return_value);
    }

  private:
    WhenAllLatch* m_latch{nullptr};
    std::exception_ptr m_exception_ptr;
    std::add_pointer_t<ReturnT> m_return_value;
};

template <>
class WhenAllTaskPromise<void>
{
  public:
    using coroutine_handle_type = std::coroutine_handle<WhenAllTaskPromise<void>>;

    WhenAllTaskPromise() noexcept = default;

    auto get_return_object() noexcept
    {
        return coroutine_handle_type::from_promise(*this);
    }

    constexpr static auto initial_suspend() noexcept -> std::suspend_always
    {
        return {};
    }

    static auto final_suspend() noexcept
    {
        struct CompletionNotifier
        {
            static constexpr auto await_ready() noexcept -> bool
            {
                return false;
            }
            static auto await_suspend(coroutine_handle_type coroutine) noexcept -> void
            {
                coroutine.promise().m_latch->notify_awaitable_completed();
            }
            static constexpr auto await_resume() noexcept -> void {}
        };

        return CompletionNotifier{};
    }

    auto unhandled_exception() noexcept -> void
    {
        m_exception_ptr = std::current_exception();
    }

    auto return_void() noexcept -> void {}

    auto result() -> void
    {
        if (m_exception_ptr)
        {
            std::rethrow_exception(m_exception_ptr);
        }
    }

    auto start(WhenAllLatch& latch) -> void
    {
        m_latch = &latch;
        coroutine_handle_type::from_promise(*this).resume();
    }

  private:
    WhenAllLatch* m_latch{nullptr};
    std::exception_ptr m_exception_ptr;
};

template <typename ReturnT>
class when_all_task
{
  public:
    // To be able to call start().
    template <typename TaskContainerT>
    friend class WhenAllReadyAwaitable;

    using promise_type          = WhenAllTaskPromise<ReturnT>;
    using coroutine_handle_type = typename promise_type::coroutine_handle_type;

    when_all_task(coroutine_handle_type coroutine) noexcept : m_coroutine(coroutine) {}

    when_all_task(const when_all_task&) = delete;
    when_all_task(when_all_task&& other) noexcept :
      m_coroutine(std::exchange(other.m_coroutine, coroutine_handle_type{}))
    {}

    auto operator=(const when_all_task&) -> when_all_task& = delete;
    auto operator=(when_all_task&&) -> when_all_task&      = delete;

    ~when_all_task()
    {
        if (m_coroutine != nullptr)
        {
            m_coroutine.destroy();
        }
    }

    auto return_value() & -> decltype(auto)
    {
        if constexpr (std::is_void_v<ReturnT>)
        {
            m_coroutine.promise().result();
            return VoidValue{};
        }
        else
        {
            return m_coroutine.promise().return_value();
        }
    }

    auto return_value() const& -> decltype(auto)
    {
        if constexpr (std::is_void_v<ReturnT>)
        {
            m_coroutine.promise().result();
            return VoidValue{};
        }
        else
        {
            return m_coroutine.promise().return_value();
        }
    }

    auto return_value() && -> decltype(auto)
    {
        if constexpr (std::is_void_v<ReturnT>)
        {
            m_coroutine.promise().result();
            return VoidValue{};
        }
        else
        {
            return m_coroutine.promise().return_value();
        }
    }

  private:
    auto start(WhenAllLatch& latch) noexcept -> void
    {
        m_coroutine.promise().start(latch);
    }

    coroutine_handle_type m_coroutine;
};

template <concepts::awaitable AwaitableT,
          typename ReturnT = typename concepts::awaitable_traits<AwaitableT&&>::awaiter_return_type>
static auto make_when_all_task(AwaitableT a) -> when_all_task<ReturnT>;

template <concepts::awaitable AwaitableT, typename ReturnT>
static auto make_when_all_task(AwaitableT a) -> when_all_task<ReturnT>
{
    if constexpr (std::is_void_v<ReturnT>)
    {
        co_await static_cast<AwaitableT&&>(a);
        co_return;
    }
    else
    {
        co_yield co_await static_cast<AwaitableT&&>(a);
    }
}

}  // namespace detail

template <concepts::awaitable... AwaitablesT>
[[nodiscard]] auto when_all(AwaitablesT... awaitables)
{
    return detail::WhenAllReadyAwaitable<
        std::tuple<detail::when_all_task<typename concepts::awaitable_traits<AwaitablesT>::awaiter_return_type>...>>(
        std::make_tuple(detail::make_when_all_task(std::move(awaitables))...));
}

template <std::ranges::range RangeT,
          concepts::awaitable AwaitableT = std::ranges::range_value_t<RangeT>,
          typename ReturnT               = typename concepts::awaitable_traits<AwaitableT>::awaiter_return_type>
[[nodiscard]] auto when_all(RangeT awaitables)
    -> detail::WhenAllReadyAwaitable<std::vector<detail::when_all_task<ReturnT>>>
{
    std::vector<detail::when_all_task<ReturnT>> output_tasks;

    // If the size is known in constant time reserve the output tasks size.
    if constexpr (std::ranges::sized_range<RangeT>)
    {
        output_tasks.reserve(std::size(awaitables));
    }

    // Wrap each task into a when_all_task.
    for (auto& a : awaitables)
    {
        output_tasks.emplace_back(detail::make_when_all_task(std::move(a)));
    }

    // Return the single awaitable that drives all the user's tasks.
    return detail::WhenAllReadyAwaitable(std::move(output_tasks));
}

}  // namespace mrc::coroutines

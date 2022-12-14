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

#include "mrc/core/std23_expected.hpp"
#include "mrc/coroutines/concepts/awaitable.hpp"
#include "mrc/coroutines/ring_buffer.hpp"

#include <concepts>
#include <coroutine>
#include <type_traits>

namespace mrc::runnable::v2 {

namespace concepts {

using namespace coroutines::concepts;

template <typename T>
concept scheduling_term = requires(T t)
{
    typename T::value_type;
    typename T::error_type;

    // explicit return_type
    requires std::same_as<typename T::return_type, std23::expected<typename T::value_type, typename T::error_type>>;

    // possible gcc/g++ bug
    // satisfaction of atomic constraint depends on itself
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=99599
    // T must be an awaitable or produce an awaitable when awaited
    requires awaitable<T> || awaitable<decltype(t.operator co_await())>;

    // the awaitable's return type must be the same as the expected return_type
    // requires std::same_as<std::decay_t<typename awaitable_traits<T>::awaiter_return_type>, typename T::return_type>;
    requires awaitable_return_same_as<T, typename T::return_type> ||
        awaitable_return_same_as<decltype(t.operator co_await()), typename T::return_type>;
};

template <typename T>
concept operator_term = requires(T t, typename T::value_type val)
{
    typename T::value_type;
    {
        t.operator co_await(std::move(val))
        } -> awaiter;
    // t.evaluate(std::move(val));
};

}  // namespace concepts

enum class CompletionType
{
};

template <typename ValueT, typename ErrorT>
struct SchedulingTerm
{
    using value_type  = ValueT;
    using error_type  = ErrorT;
    using return_type = std23::expected<value_type, error_type>;
};

struct Done
{};

class TestSchedulingTerm : public SchedulingTerm<int, int>
{
    struct Awaiter
    {
        constexpr Awaiter(TestSchedulingTerm& scheduling_term) : m_scheduling_term(scheduling_term) {}

        constexpr static bool await_ready() noexcept
        {
            return true;
        }

        constexpr static void await_suspend(std::coroutine_handle<> handle) noexcept {}

        return_type await_resume()
        {
            return {++(m_scheduling_term.m_value)};
        }

        TestSchedulingTerm& m_scheduling_term;
    };

  public:
    using value_type  = int;
    using error_type  = int;
    using return_type = std23::expected<value_type, error_type>;

    [[nodiscard]] Awaiter operator co_await()
    {
        return Awaiter{*this};
    }

  private:
    int m_value{0};
    friend class Awaiter;
};

static_assert(concepts::scheduling_term<TestSchedulingTerm>);

template <typename T>
class STRB : public SchedulingTerm<T, coroutines::RingBufferOpStatus>
{
  public:
    [[nodiscard]] auto operator co_await()
    {
        return m_channel->read();
    }

    std::shared_ptr<coroutines::RingBuffer<T>> m_channel;
};

class SchedulingTask : public SchedulingTerm<int, Done>
{
  public:
    using base_type = SchedulingTerm<int, Done>;

    SchedulingTask()
    {
        m_task_fn = []() -> coroutines::Task<typename base_type::return_type> {
            co_return std23::expected<int, Done>{42};
        };
    }
    [[nodiscard]] auto operator co_await() const noexcept
    {
        return m_task_fn();
    }

    std::function<coroutines::Task<typename base_type::return_type>()> m_task_fn;
};

static_assert(concepts::awaitable<coroutines::Task<int>>);

static_assert(concepts::scheduling_term<TestSchedulingTerm>);
static_assert(concepts::scheduling_term<STRB<int>>);
static_assert(concepts::scheduling_term<SchedulingTask>);

struct A
{};

static std23::expected<A, int> foo(int i = 0)
{
    if (i != 0)
    {
        return std23::unexpected<int>(i);
    }

    return {};
}

// coroutines::Task<void> make_runnable()
// {
//     return []() -> coroutines::Task<void> {
//         do
//         {
//             auto status = co_await (co_await scheduling_term()).and_then(operator_term);
//         } while(status);
//     }();
// }

}  // namespace mrc::runnable::v2

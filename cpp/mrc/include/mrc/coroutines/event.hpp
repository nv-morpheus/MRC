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

// original source code modified to include originators copyright,
// adapt the code for use with MRC

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

#include "mrc/coroutines/concepts/executor.hpp"
#include "mrc/coroutines/thread_local_context.hpp"

#include <atomic>
#include <coroutine>

namespace mrc::coroutines {

enum class ResumeOrderPolicy
{
    /// Last in first out, this is the default policy and will execute the fastest
    /// if you do not need the first waiter to execute first upon the event being set.
    lifo,
    /// First in first out, this policy has an extra overhead to reverse the order of
    /// the waiters but will guarantee the ordering is fifo.
    fifo
};

/**
 * Event is a manully triggered thread safe signal that can be co_await()'ed by multiple awaiters.
 * Each awaiter should co_await the event and upon the event being set each awaiter will have their
 * coroutine resumed.
 *
 * The event can be manually reset to the un-set state to be re-used.
 * \code
t1: coroutines::event e;
...
t2: func(coroutines::event& e) { ... co_await e; ... }
...
t1: do_work();
t1: e.set();
...
t2: resume()
 * \endcode
 */
class Event
{
  public:
    struct Awaiter : ThreadLocalContext
    {
        /**
         * @param e The event to wait for it to be set.
         */
        Awaiter(const Event& e) noexcept : m_event(e) {}

        /**
         * @return True if the event is already set, otherwise false to suspend this coroutine.
         */
        auto await_ready() const noexcept -> bool
        {
            return m_event.is_set();
        }

        /**
         * Adds this coroutine to the list of awaiters in a thread safe fashion.  If the event
         * is set while attempting to add this coroutine to the awaiters then this will return false
         * to resume execution immediately.
         * @return False if the event is already set, otherwise true to suspend this coroutine.
         */
        auto await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> bool;

        /**
         * Nothing to do on resume.
         */
        auto await_resume() noexcept -> void;

        /**
         * @brief Used to resume the coroutine
         */
        void resume();

        /// Refernce to the event that this awaiter is waiting on.
        const Event& m_event;
        /// The awaiting continuation coroutine handle.
        std::coroutine_handle<> m_awaiting_coroutine;
        /// The next awaiter in line for this event, nullptr if this is the end.
        Awaiter* m_next{nullptr};
    };

    /**
     * Creates an event with the given initial state of being set or not set.
     * @param initially_set By default all events start as not set, but if needed this parameter can
     *                      set the event to already be triggered.
     */
    explicit Event(bool initially_set = false) noexcept;
    ~Event() = default;

    Event(const Event&)                    = delete;
    Event(Event&&)                         = delete;
    auto operator=(const Event&) -> Event& = delete;
    auto operator=(Event&&) -> Event&      = delete;

    /**
     * @return True if this event is currently in the set state.
     */
    auto is_set() const noexcept -> bool
    {
        return m_state.load(std::memory_order_acquire) == this;
    }

    /**
     * Sets this event and resumes all awaiters.  Note that all waiters will be resumed onto this
     * thread of execution.
     * @param policy The order in which the waiters should be resumed, defaults to LIFO since it
     *               is more efficient, FIFO requires reversing the order of the waiters first.
     */
    auto set(ResumeOrderPolicy policy = ResumeOrderPolicy::lifo) noexcept -> void;

    /**
     * Sets this event and resumes all awaiters onto the given executor.  This will distribute
     * the waiters across the executor's threads.
     */
    template <concepts::executor ExecutorT>
    auto set(ExecutorT& e, ResumeOrderPolicy policy = ResumeOrderPolicy::lifo) noexcept -> void
    {
        void* old_value = m_state.exchange(this, std::memory_order::acq_rel);
        if (old_value != this)
        {
            // If FIFO has been requsted then reverse the order upon resuming.
            if (policy == ResumeOrderPolicy::fifo)
            {
                old_value = reverse(static_cast<Awaiter*>(old_value));
            }
            // else lifo nothing to do

            auto* waiters = static_cast<Awaiter*>(old_value);
            while (waiters != nullptr)
            {
                auto* next = waiters->m_next;
                e.resume(waiters->m_awaiting_coroutine);
                waiters = next;
            }
        }
    }

    /**
     * @return An awaiter struct to suspend and resume this coroutine for when the event is set.
     */
    auto operator co_await() const noexcept -> Awaiter
    {
        return {*this};
    }

    /**
     * Resets the event from set to not set so it can be re-used.  If the event is not currently
     * set then this function has no effect.
     */
    auto reset() noexcept -> void;

  protected:
    /// For access to m_state.
    friend struct Awaiter;
    /// The state of the event, nullptr is not set with zero awaiters.  Set to an awaiter* there are
    /// coroutines awaiting the event to be set, and set to this the event has triggered.
    /// 1) nullptr == not set
    /// 2) awaiter* == linked list of awaiters waiting for the event to trigger.
    /// 3) this == The event is triggered and all awaiters are resumed.
    mutable std::atomic<void*> m_state;

  private:
    /**
     * Reverses the set of waiters from LIFO->FIFO and returns the new head.
     */
    static auto reverse(Awaiter* curr) -> Awaiter*;
};

}  // namespace mrc::coroutines

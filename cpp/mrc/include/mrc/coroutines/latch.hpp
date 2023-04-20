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

#include "mrc/coroutines/event.hpp"
#include "mrc/coroutines/thread_pool.hpp"

#include <atomic>

namespace mrc::coroutines {
/**
 * The latch is thread safe counter to wait for 1 or more other tasks to complete, they signal their
 * completion by calling `count_down()` on the latch and upon the latch counter reaching zero the
 * coroutine `co_await`ing the latch then resumes execution.
 *
 * This is useful for spawning many worker tasks to complete either a computationally complex task
 * across a thread pool of workers, or waiting for many asynchronous results like http requests
 * to complete.
 */
class Latch
{
  public:
    /**
     * Creates a latch with the given count of tasks to wait to complete.
     * @param count The number of tasks to wait to complete, if this is zero or negative then the
     *              latch starts 'completed' immediately and execution is resumed with no suspension.
     */
    Latch(std::ptrdiff_t count) noexcept : m_count(count), m_event(count <= 0) {}

    Latch(const Latch&)                        = delete;
    Latch(Latch&&) noexcept                    = delete;
    auto operator=(const Latch&) -> Latch&     = delete;
    auto operator=(Latch&&) noexcept -> Latch& = delete;

    /**
     * @return True if the latch has been counted down to zero.
     */
    auto is_ready() const noexcept -> bool
    {
        return m_event.is_set();
    }

    /**
     * @return The number of tasks this latch is still waiting to complete.
     */
    auto remaining() const noexcept -> std::size_t
    {
        return m_count.load(std::memory_order::acquire);
    }

    /**
     * If the latch counter goes to zero then the task awaiting the latch is resumed.
     * @param n The number of tasks to complete towards the latch, defaults to 1.
     */
    auto count_down(std::ptrdiff_t n = 1) noexcept -> void
    {
        if (m_count.fetch_sub(n, std::memory_order::acq_rel) <= n)
        {
            m_event.set();
        }
    }

    /**
     * If the latch counter goes to then the task awaiting the latch is resumed on the given
     * thread pool.
     * @param tp The thread pool to schedule the task that is waiting on the latch on.
     * @param n The number of tasks to complete towards the latch, defaults to 1.
     */
    auto count_down(ThreadPool& tp, std::ptrdiff_t n = 1) noexcept -> void
    {
        if (m_count.fetch_sub(n, std::memory_order::acq_rel) <= n)
        {
            m_event.set(tp);
        }
    }

    auto operator co_await() const noexcept -> Event::Awaiter
    {
        return m_event.operator co_await();
    }

  private:
    /// The number of tasks to wait for completion before triggering the event to resume.
    std::atomic<std::ptrdiff_t> m_count;
    /// The event to trigger when the latch counter reaches zero, this resume the coroutine that
    /// is co_await'ing on the latch.
    Event m_event;
};

}  // namespace mrc::coroutines

/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/coroutines/scheduler.hpp"
#include "mrc/coroutines/task.hpp"

#include <chrono>
#include <coroutine>
#include <queue>
#include <utility>
#include <vector>

#pragma once

namespace mrc::coroutines {

class TestScheduler : public Scheduler
{
  private:
    struct Operation
    {
      public:
        Operation(TestScheduler* self, std::chrono::time_point<std::chrono::steady_clock> time);

        static constexpr bool await_ready()
        {
            return false;
        }

        void await_suspend(std::coroutine_handle<> handle);

        void await_resume() {}

      private:
        TestScheduler* m_self;
        std::chrono::time_point<std::chrono::steady_clock> m_time;
    };

    using item_t = std::pair<std::coroutine_handle<>, std::chrono::time_point<std::chrono::steady_clock>>;
    struct ItemCompare
    {
        bool operator()(item_t& lhs, item_t& rhs);
    };

    std::priority_queue<item_t, std::vector<item_t>, ItemCompare> m_queue;
    std::chrono::time_point<std::chrono::steady_clock> m_time = std::chrono::steady_clock::now();

  public:
    /**
     * @brief Enqueue's the coroutine handle to be resumed at the current logical time.
     */
    void resume(std::coroutine_handle<> handle) noexcept override;

    /**
     * Suspends the current function and enqueue's it to be resumed at the current logical time.
     */
    mrc::coroutines::Task<> yield() override;

    /**
     * Suspends the current function and enqueue's it to be resumed at the current logica time + the given duration.
     */
    mrc::coroutines::Task<> yield_for(std::chrono::milliseconds time) override;

    /**
     * Suspends the current function and enqueue's it to be resumed at the given logical time.
     */
    mrc::coroutines::Task<> yield_until(std::chrono::time_point<std::chrono::steady_clock> time) override;

    /**
     * Returns the time according to the scheduler. Time may be progressed by resume_next, resume_for, and resume_until.
     *
     * @return the current time according to the scheduler.
     */
    std::chrono::time_point<std::chrono::steady_clock> time();

    /**
     * Immediately resumes the next-in-queue coroutine handle.
     *
     *  @return true if more coroutines exist in the queue after resuming, false otherwise.
     */
    bool resume_next();

    /**
     * Immediately resumes next-in-queue coroutines up to the current logical time + the given duration, in-order.
     *
     *  @return true if more coroutines exist in the queue after resuming, false otherwise.
     */
    bool resume_for(std::chrono::milliseconds time);

    /**
     * Immediately resumes next-in-queue coroutines up to the given logical time.
     *
     *  @return true if more coroutines exist in the queue after resuming, false otherwise.
     */
    bool resume_until(std::chrono::time_point<std::chrono::steady_clock> time);
};

}  // namespace mrc::coroutines

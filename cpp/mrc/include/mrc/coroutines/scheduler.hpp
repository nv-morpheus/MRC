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

#pragma once

#include "mrc/coroutines/task.hpp"
#include "mrc/coroutines/task_container.hpp"

#include <coroutine>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>

namespace mrc::coroutines {

/**
 * @brief Scheduler base class
 *
 * Allows all schedulers to be discovered via the mrc::this_thread::current_scheduler()
 */
class Scheduler : public std::enable_shared_from_this<Scheduler>
{
  public:
    struct Operation
    {
        Operation(Scheduler& scheduler);

        constexpr static auto await_ready() noexcept -> bool
        {
            return false;
        }

        std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept;

        constexpr static auto await_resume() noexcept -> void {}

        Scheduler& m_scheduler;
        std::coroutine_handle<> m_awaiting_coroutine;
        Operation* m_next{nullptr};
    };

    Scheduler()          = default;
    virtual ~Scheduler() = default;

    /**
     * @brief Description of Scheduler
     */
    virtual const std::string& description() const = 0;

    /**
     * Schedules the currently executing coroutine to be run on this thread pool.  This must be
     * called from within the coroutines function body to schedule the coroutine on the thread pool.
     * @throw std::runtime_error If the thread pool is `shutdown()` scheduling new tasks is not permitted.
     * @return The operation to switch from the calling scheduling thread to the executor thread
     *         pool thread.
     */
    [[nodiscard]] virtual auto schedule() -> Operation;

    // Enqueues a message without waiting for it. Must return void since the caller will not get the return value
    virtual TaskContainer::StartOperation schedule(Task<void>&& task);

    /**
     * Schedules any coroutine handle that is ready to be resumed.
     * @param handle The coroutine handle to schedule.
     */
    virtual auto resume(std::coroutine_handle<> coroutine) -> void = 0;

    /**
     * @brief Runs the selected task until it is complete. Blocks the current thread and may optionally use that thread
     * for execution.
     *
     * @param task The task to run.
     */
    virtual void run_until_complete(Task<void> task);

    [[nodiscard]] auto yield() -> Operation;

    /**
     * If the calling thread controlled by a Scheduler, return a pointer to the Scheduler
     */
    static auto from_current_thread() noexcept -> Scheduler*;

    /**
     * If the calling thread is owned by a thread_pool, return the thread index (rank) of the current thread with
     * respect the threads in the pool; otherwise, return the std::hash of std::this_thread::get_id
     */
    static auto get_thread_id() noexcept -> std::size_t;

  protected:
    virtual auto on_thread_start(std::size_t) -> void;

    TaskContainer& get_task_container() const;

  private:
    virtual std::unique_ptr<TaskContainer> make_task_container() const;

    virtual std::coroutine_handle<> schedule_operation(Operation* operation) = 0;

    mutable std::mutex m_mutex;

    std::unique_ptr<TaskContainer> m_task_container;

    thread_local static Scheduler* m_thread_local_scheduler;
    thread_local static std::size_t m_thread_id;
};

}  // namespace mrc::coroutines

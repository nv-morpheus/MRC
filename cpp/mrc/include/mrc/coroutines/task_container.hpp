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

#include "mrc/coroutines/task.hpp"

#include <atomic>
#include <coroutine>
#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace mrc::coroutines {
class Scheduler;

class TaskContainer
{
  public:
    enum class GarbageCollectPolicy
    {
        /// Execute garbage collection.
        yes,
        /// Do not execute garbage collection.
        no
    };

    struct StartOperation
    {
        StartOperation(TaskContainer& parent, Task<void>&& task, GarbageCollectPolicy cleanup);

        constexpr static auto await_ready() noexcept -> bool
        {
            return false;
        }

        std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiting_coroutine);

        constexpr static auto await_resume() noexcept -> void {}

        TaskContainer& m_parent;
        Task<void> m_task;
        GarbageCollectPolicy m_cleanup;

        std::coroutine_handle<> m_awaiting_coroutine;
    };

    using task_position_t = std::list<std::optional<Task<>>>::iterator;

    /**
     * @param e Tasks started in the container are scheduled onto this executor.  For tasks created
     *           from a coro::io_scheduler, this would usually be that coro::io_scheduler instance.
     * @param opts Task container options.
     */
    TaskContainer(std::shared_ptr<Scheduler> e, std::size_t concurrency = 1024);
    TaskContainer(const TaskContainer&)                    = delete;
    TaskContainer(TaskContainer&&)                         = delete;
    auto operator=(const TaskContainer&) -> TaskContainer& = delete;
    auto operator=(TaskContainer&&) -> TaskContainer&      = delete;
    ~TaskContainer();

    /**
     * Stores a user task and starts its execution on the container's thread pool.
     * @param user_task The scheduled user's task to store in this task container and start its execution.
     * @param cleanup Should the task container run garbage collect at the beginning of this store
     *                call?  Calling at regular intervals will reduce memory usage of completed
     *                tasks and allow for the task container to re-use allocated space.
     */
    StartOperation start(Task<void>&& user_task, GarbageCollectPolicy cleanup = GarbageCollectPolicy::yes);

    /**
     * Garbage collects any tasks that are marked as deleted.  This frees up space to be re-used by
     * the task container for newly stored tasks.
     * @return The number of tasks that were deleted.
     */
    auto garbage_collect() -> std::size_t;  // __attribute__((used))

    /**
     * @return The number of tasks that are awaiting deletion.
     */
    auto delete_task_size() const -> std::size_t;

    /**
     * @return True if there are no tasks awaiting deletion.
     */
    auto delete_tasks_empty() const -> bool;

    /**
     * @return The number of active tasks in the container.
     */
    auto size() const -> std::size_t;

    /**
     * @return True if there are no active tasks in the container.
     */
    auto empty() const -> bool;

    /**
     * Will continue to garbage collect and yield until all tasks are complete.  This method can be
     * co_await'ed to make it easier to wait for the task container to have all its tasks complete.
     *
     * This does not shut down the task container, but can be used when shutting down, or if your
     * logic requires all the tasks contained within to complete, it is similar to coro::latch.
     */
    auto garbage_collect_and_yield_until_empty() -> Task<void>;

  private:
    /**
     * Interal GC call, expects the public function to lock.
     */
    auto gc_internal() -> std::size_t;

    /**
     * Encapsulate the users tasks in a cleanup task which marks itself for deletion upon
     * completion.  Simply co_await the users task until its completed and then mark the given
     * position within the task manager as being deletable.  The scheduler's next iteration
     * in its event loop will then free that position up to be re-used.
     *
     * This function will also unconditionally catch all unhandled exceptions by the user's
     * task to prevent the scheduler from throwing exceptions.
     * @param user_task The user's task.
     * @param pos The position where the task data will be stored in the task manager.
     * @return The user's task wrapped in a self cleanup task.
     */
    auto make_cleanup_task(Task<void> user_task, task_position_t pos) -> Task<void>;

    /**
     * @brief Starts the task associated with the provided operator.
     *
     * @param lock Move a lock into this function to prevent locking and relocking issues
     * @param op Operator to start
     * @return std::coroutine_handle<> Returns a handle to the operator's coroutine which should be `resume()`. This
     * value is returned instead of called directly to allow it to be used as the return value in await_suspend()
     */
    virtual auto do_start(std::unique_lock<std::mutex>&& lock, StartOperation* op) -> std::coroutine_handle<>;

    /**
     * @brief Implements the scheduling of the start() call. Can possibly block depending on the number of outstanding
     * tasks.
     *
     * @param op The awaitable operation to schedule
     * @return std::coroutine_handle<> Returns a handle to the operator's coroutine which should be `resume()`. This is
     * returned instead of called directly to allow it to be used as the return value in await_suspend()
     */
    virtual std::coroutine_handle<> schedule_start_operation(StartOperation* op);

    /// Mutex for safely mutating the task containers across threads, expected usage is within
    /// thread pools for indeterminate lifetime requests.
    std::mutex m_mutex{};
    /// The max number of concurrent tasks
    std::size_t m_concurrency{1024};
    /// The number of alive tasks.
    std::atomic<std::size_t> m_size{};
    /// Maintains the lifetime of the tasks until they are completed.
    std::list<std::optional<Task<void>>> m_tasks{};
    /// The set of tasks that have completed and need to be deleted.
    std::vector<task_position_t> m_tasks_to_delete{};
    /// The current free position within the task indexes list.  Anything before
    std::shared_ptr<Scheduler> m_scheduler{nullptr};

    std::list<StartOperation*> m_waiting_start_operations;
};

}  // namespace mrc::coroutines

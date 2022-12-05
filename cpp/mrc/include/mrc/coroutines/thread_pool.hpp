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

#include "mrc/coroutines/concepts/range_of.hpp"
#include "mrc/coroutines/task.hpp"
#include "mrc/coroutines/thread_local_context.hpp"

#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <ranges>
#include <thread>
#include <variant>
#include <vector>

namespace mrc::coroutines {
/**
 * Creates a thread pool that executes arbitrary coroutine tasks in a FIFO scheduler policy.
 * The thread pool by default will create an execution thread per available core on the system.
 *
 * When shutting down, either by the thread pool destructing or by manually calling shutdown()
 * the thread pool will stop accepting new tasks but will complete all tasks that were scheduled
 * prior to the shutdown request.
 */
class ThreadPool
{
  public:
    /**
     * An operation is an awaitable type with a coroutine to resume the task scheduled on one of
     * the executor threads.
     */
    class Operation  // : ThreadLocalContext
    {
        friend class ThreadPool;
        /**
         * Only thread_pools can create operations when a task is being scheduled.
         * @param tp The thread pool that created this operation.
         */
        explicit Operation(ThreadPool& tp) noexcept;

      public:
        /**
         * Operations always pause so the executing thread can be switched.
         */
        constexpr static auto await_ready() noexcept -> bool
        {
            return false;
        }

        /**
         * Suspending always returns to the caller (using void return of await_suspend()) and
         * stores the coroutine internally for the executing thread to resume from.
         * Capture any thread-local state from the caller so it can be resumed on a thread from the pool.
         */
        auto await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> void;

        /**
         * this is the function called first by the thread pool's executing thread.
         * resume any thread local state that was captured on suspend
         */
        auto await_resume() noexcept -> void;

      private:
        /// The thread pool that this operation will execute on.
        ThreadPool& m_thread_pool;
        /// The coroutine awaiting execution.
        std::coroutine_handle<> m_awaiting_coroutine{nullptr};
        /// Span to measure time spent being scheduled
        // srf_OTEL_TRACE(trace::Handle<trace::Span> m_span{nullptr});
    };

    struct Options
    {
        /// The number of executor threads for this thread pool.  Uses the hardware concurrency
        /// value by default.
        uint32_t thread_count = std::thread::hardware_concurrency();
        /// Functor to call on each executor thread upon starting execution.  The parameter is the
        /// thread's ID assigned to it by the thread pool.
        std::function<void(std::size_t)> on_thread_start_functor = nullptr;
        /// Functor to call on each executor thread upon stopping execution.  The parameter is the
        /// thread's ID assigned to it by the thread pool.
        std::function<void(std::size_t)> on_thread_stop_functor = nullptr;
        /// Description
        std::string description;
    };

    /**
     * @param opts Thread pool configuration options.
     */
    explicit ThreadPool(Options opts = Options{.thread_count            = std::thread::hardware_concurrency(),
                                               .on_thread_start_functor = nullptr,
                                               .on_thread_stop_functor  = nullptr});

    ThreadPool(const ThreadPool&)                    = delete;
    ThreadPool(ThreadPool&&)                         = delete;
    auto operator=(const ThreadPool&) -> ThreadPool& = delete;
    auto operator=(ThreadPool&&) -> ThreadPool&      = delete;

    virtual ~ThreadPool();

    /**
     * @return The number of executor threads for processing tasks.
     */
    auto thread_count() const noexcept -> uint32_t
    {
        return m_threads.size();
    }

    /**
     * Schedules the currently executing coroutine to be run on this thread pool.  This must be
     * called from within the coroutines function body to schedule the coroutine on the thread pool.
     * @throw std::runtime_error If the thread pool is `shutdown()` scheduling new tasks is not permitted.
     * @return The operation to switch from the calling scheduling thread to the executor thread
     *         pool thread.
     */
    [[nodiscard]] auto schedule() -> Operation;

    /**
     * @throw std::runtime_error If the thread pool is `shutdown()` scheduling new tasks is not permitted.
     * @param f The function to execute on the thread pool.
     * @param args The arguments to call the functor with.
     * @return A task that wraps the given functor to be executed on the thread pool.
     */
    template <typename FunctorT, typename... ArgumentsT>
    [[nodiscard]] auto enqueue(FunctorT&& f, ArgumentsT... args) -> Task<decltype(f(std::forward<ArgumentsT>(args)...))>
    {
        co_await schedule();

        if constexpr (std::is_same_v<void, decltype(f(std::forward<ArgumentsT>(args)...))>)
        {
            f(std::forward<ArgumentsT>(args)...);
            co_return;
        }
        else
        {
            co_return f(std::forward<ArgumentsT>(args)...);
        }
    }

    /**
     * Schedules any coroutine handle that is ready to be resumed.
     * @param handle The coroutine handle to schedule.
     */
    auto resume(std::coroutine_handle<> handle) noexcept -> void;

    /**
     * Schedules the set of coroutine handles that are ready to be resumed.
     * @param handles The coroutine handles to schedule.
     */
    template <concepts::range_of<std::coroutine_handle<>> RangeT>
    auto resume(const RangeT& handles) noexcept -> void
    {
        m_size.fetch_add(std::size(handles), std::memory_order::release);

        size_t null_handles{0};

        {
            std::scoped_lock lk{m_wait_mutex};
            for (const auto& handle : handles)
            {
                if (handle != nullptr) [[likely]]
                {
                    m_queue.emplace_back(handle);
                }
                else
                {
                    ++null_handles;
                }
            }
        }

        if (null_handles > 0)
        {
            m_size.fetch_sub(null_handles, std::memory_order::release);
        }

        m_wait_cv.notify_one();
    }

    /**
     * Immediately yields the current task and places it at the end of the queue of tasks waiting
     * to be processed.  This will immediately be picked up again once it naturally goes through the
     * FIFO task queue.  This function is useful to yielding long processing tasks to let other tasks
     * get processing time.
     */
    [[nodiscard]] auto yield() -> Operation
    {
        return schedule();
    }

    /**
     * Shutsdown the thread pool.  This will finish any tasks scheduled prior to calling this
     * function but will prevent the thread pool from scheduling any new tasks.  This call is
     * blocking and will wait until all inflight tasks are completed before returnin.
     */
    auto shutdown() noexcept -> void;

    /**
     * @return The number of tasks waiting in the task queue + the executing tasks.
     */
    auto size() const noexcept -> std::size_t
    {
        return m_size.load(std::memory_order::acquire);
    }

    /**
     * @return True if the task queue is empty and zero tasks are currently executing.
     */
    auto empty() const noexcept -> bool
    {
        return size() == 0;
    }

    /**
     * @return The number of tasks waiting in the task queue to be executed.
     */
    auto queue_size() const noexcept -> std::size_t
    {
        // Might not be totally perfect but good enough, avoids acquiring the lock for now.
        std::atomic_thread_fence(std::memory_order::acquire);
        return m_queue.size();
    }

    /**
     * @return True if the task queue is currently empty.
     */
    auto queue_empty() const noexcept -> bool
    {
        return queue_size() == 0;
    }

    /**
     * If the calling thread is owned by a thread_pool, return a pointer to the thread_pool; otherwise, return a
     * nullptr;
     */
    static auto from_current_thread() -> ThreadPool*;

    /**
     * If the calling thread is owned by a thread_pool, return the thread index (rank) of the current thread with
     * respect the threads in the pool; otherwise, return the std::hash of std::this_thread::get_id
     */
    static auto get_thread_id() -> std::size_t;

    /**
     * @return std::string description of the thread pool
     */
    const std::string& description() const;

  private:
    /// The configuration options.
    Options m_opts;
    /// The background executor threads.
    std::vector<std::jthread> m_threads;

    /// Mutex for executor threads to sleep on the condition variable.
    std::mutex m_wait_mutex;
    /// Condition variable for each executor thread to wait on when no tasks are available.
    std::condition_variable_any m_wait_cv;
    /// FIFO queue of tasks waiting to be executed.
    std::deque<std::coroutine_handle<>> m_queue;
    /**
     * Each background thread runs from this function.
     * @param stop_token Token which signals when shutdown() has been called.
     * @param idx The executor's idx for internal data structure accesses.
     */
    auto executor(std::stop_token stop_token, std::size_t idx) -> void;

    /**
     * @param handle Schedules the given coroutine to be executed upon the first available thread.
     */
    auto schedule_impl(std::coroutine_handle<> handle) noexcept -> void;

    /// The number of tasks in the queue + currently executing.
    std::atomic<std::size_t> m_size{0};
    /// Has the thread pool been requested to shut down?
    std::atomic<bool> m_shutdown_requested{false};

    /// thead local pointer to the owning thread pool
    static thread_local ThreadPool* m_self;

    /// user defined description
    std::string m_description;

    /// thread local index of worker thread
    static thread_local std::size_t m_thread_id;
};

}  // namespace mrc::coroutines

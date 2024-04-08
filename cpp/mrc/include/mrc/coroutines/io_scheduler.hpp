/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/coroutines/detail/poll_info.hpp"
#include "mrc/coroutines/fd.hpp"
#include "mrc/coroutines/scheduler.hpp"
#include "mrc/coroutines/task.hpp"
#include "mrc/coroutines/thread_pool.hpp"
#include "mrc/coroutines/time.hpp"

#ifdef LIBCORO_FEATURE_NETWORKING
    #include "coro/net/socket.hpp"
#endif

#include <sys/eventfd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace mrc::coroutines {
enum class PollOperation : uint64_t;
enum class PollStatus;

class IoScheduler : public Scheduler
{
  private:
    using timed_events_t = detail::PollInfo::timed_events_t;

  public:
    static std::shared_ptr<IoScheduler> get_instance();

    class schedule_operation;

    friend schedule_operation;

    enum class ThreadStrategy
    {
        /// Spawns a dedicated background thread for the scheduler to run on.
        spawn,
        /// Requires the user to call process_events() to drive the scheduler.
        manual
    };

    enum class ExecutionStrategy
    {
        /// Tasks will be FIFO queued to be executed on a thread pool.  This is better for tasks that
        /// are long lived and will use lots of CPU because long lived tasks will block other i/o
        /// operations while they complete.  This strategy is generally better for lower latency
        /// requirements at the cost of throughput.
        process_tasks_on_thread_pool,
        /// Tasks will be executed inline on the io scheduler thread.  This is better for short tasks
        /// that can be quickly processed and not block other i/o operations for very long.  This
        /// strategy is generally better for higher throughput at the cost of latency.
        process_tasks_inline
    };

    struct Options
    {
        /// Should the io scheduler spawn a dedicated event processor?
        ThreadStrategy thread_strategy{ThreadStrategy::spawn};
        /// If spawning a dedicated event processor a functor to call upon that thread starting.
        std::function<void()> on_io_thread_start_functor{nullptr};
        /// If spawning a dedicated event processor a functor to call upon that thread stopping.
        std::function<void()> on_io_thread_stop_functor{nullptr};
        /// Thread pool options for the task processor threads.  See thread pool for more details.
        ThreadPool::Options pool{
            .thread_count = ((std::thread::hardware_concurrency() > 1) ? (std::thread::hardware_concurrency() - 1) : 1),
            .on_thread_start_functor = nullptr,
            .on_thread_stop_functor  = nullptr};

        /// If inline task processing is enabled then the io worker will resume tasks on its thread
        /// rather than scheduling them to be picked up by the thread pool.
        const ExecutionStrategy execution_strategy{ExecutionStrategy::process_tasks_on_thread_pool};
    };

    explicit IoScheduler(Options opts = Options{
                             .thread_strategy            = ThreadStrategy::spawn,
                             .on_io_thread_start_functor = nullptr,
                             .on_io_thread_stop_functor  = nullptr,
                             .pool                       = {.thread_count            = ((std::thread::hardware_concurrency() > 1)
                                                                                            ? (std::thread::hardware_concurrency() - 1)
                                                                                            : 1),
                                                            .on_thread_start_functor = nullptr,
                                                            .on_thread_stop_functor  = nullptr},
                             .execution_strategy         = ExecutionStrategy::process_tasks_on_thread_pool});

    IoScheduler(const IoScheduler&)                    = delete;
    IoScheduler(IoScheduler&&)                         = delete;
    auto operator=(const IoScheduler&) -> IoScheduler& = delete;
    auto operator=(IoScheduler&&) -> IoScheduler&      = delete;

    ~IoScheduler() override;

    /**
     * Given a ThreadStrategy::manual this function should be called at regular intervals to
     * process events that are ready.  If a using ThreadStrategy::spawn this is run continously
     * on a dedicated background thread and does not need to be manually invoked.
     * @param timeout If no events are ready how long should the function wait for events to be ready?
     *                Passing zero (default) for the timeout will check for any events that are
     *                ready now, and then return.  This could be zero events.  Passing -1 means block
     *                indefinitely until an event happens.
     * @param return The number of tasks currently executing or waiting to execute.
     */
    auto process_events(std::chrono::milliseconds timeout = std::chrono::milliseconds{0}) -> std::size_t;

    class schedule_operation
    {
        friend class IoScheduler;
        explicit schedule_operation(IoScheduler& scheduler) noexcept : m_scheduler(scheduler) {}

      public:
        /**
         * Operations always pause so the executing thread can be switched.
         */
        static constexpr auto await_ready() noexcept -> bool
        {
            return false;
        }

        /**
         * Suspending always returns to the caller (using void return of await_suspend()) and
         * stores the coroutine internally for the executing thread to resume from.
         */
        auto await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept -> void
        {
            if (m_scheduler.m_opts.execution_strategy == ExecutionStrategy::process_tasks_inline)
            {
                m_scheduler.m_size.fetch_add(1, std::memory_order::release);
                {
                    std::scoped_lock lk{m_scheduler.m_scheduled_tasks_mutex};
                    m_scheduler.m_scheduled_tasks.emplace_back(awaiting_coroutine);
                }

                // Trigger the event to wake-up the scheduler if this event isn't currently triggered.
                bool expected{false};
                if (m_scheduler.m_schedule_fd_triggered.compare_exchange_strong(expected,
                                                                                true,
                                                                                std::memory_order::release,
                                                                                std::memory_order::relaxed))
                {
                    eventfd_t value{1};
                    eventfd_write(m_scheduler.m_schedule_fd, value);
                }
            }
            else
            {
                m_scheduler.m_thread_pool->resume(awaiting_coroutine);
            }
        }

        /**
         * no-op as this is the function called first by the thread pool's executing thread.
         */
        auto await_resume() noexcept -> void {}

      private:
        /// The thread pool that this operation will execute on.
        IoScheduler& m_scheduler;
    };

    /**
     * Schedules the current task onto this IoScheduler for execution.
     */
    auto schedule() -> schedule_operation
    {
        return schedule_operation{*this};
    }

    /**
     * Schedules a task onto the IoScheduler and moves ownership of the task to the IoScheduler.
     * Only void return type tasks can be scheduled in this manner since the task submitter will no
     * longer have control over the scheduled task.
     * @param task The task to execute on this IoScheduler.  It's lifetime ownership will be transferred
     *             to this IoScheduler.
     */
    auto schedule(mrc::coroutines::Task<void>&& task) -> void;

    /**
     * Schedules the current task to run after the given amount of time has elapsed.
     * @param amount The amount of time to wait before resuming execution of this task.
     *               Given zero or negative amount of time this behaves identical to schedule().
     */
    [[nodiscard]] auto schedule_after(std::chrono::milliseconds amount) -> mrc::coroutines::Task<void>;

    /**
     * Schedules the current task to run at a given time point in the future.
     * @param time The time point to resume execution of this task.  Given 'now' or a time point
     *             in the past this behaves identical to schedule().
     */
    [[nodiscard]] auto schedule_at(time_point_t time) -> mrc::coroutines::Task<void>;

    /**
     * Yields the current task to the end of the queue of waiting tasks.
     */
    [[nodiscard]] mrc::coroutines::Task<void> yield() override
    {
        co_await schedule_operation{*this};
    };

    /**
     * Yields the current task for the given amount of time.
     * @param amount The amount of time to yield for before resuming executino of this task.
     *               Given zero or negative amount of time this behaves identical to yield().
     */
    [[nodiscard]] mrc::coroutines::Task<void> yield_for(std::chrono::milliseconds amount) override;

    /**
     * Yields the current task until the given time point in the future.
     * @param time The time point to resume execution of this task.  Given 'now' or a time point in the
     *             in the past this behaves identical to yield().
     */
    [[nodiscard]] mrc::coroutines::Task<void> yield_until(time_point_t time) override;

    /**
     * Polls the given file descriptor for the given operations.
     * @param fd The file descriptor to poll for events.
     * @param op The operations to poll for.
     * @param timeout The amount of time to wait for the events to trigger.  A timeout of zero will
     *                block indefinitely until the event triggers.
     * @return The result of the poll operation.
     */
    [[nodiscard]] auto poll(fd_t fd,
                            mrc::coroutines::PollOperation op,
                            std::chrono::milliseconds timeout = std::chrono::milliseconds{0})
        -> mrc::coroutines::Task<PollStatus>;

#ifdef LIBCORO_FEATURE_NETWORKING
    /**
     * Polls the given mrc::coroutines::net::socket for the given operations.
     * @param sock The socket to poll for events on.
     * @param op The operations to poll for.
     * @param timeout The amount of time to wait for the events to trigger.  A timeout of zero will
     *                block indefinitely until the event triggers.
     * @return THe result of the poll operation.
     */
    [[nodiscard]] auto poll(const net::socket& sock,
                            mrc::coroutines::poll_op op,
                            std::chrono::milliseconds timeout = std::chrono::milliseconds{0})
        -> mrc::coroutines::Task<poll_status>
    {
        return poll(sock.native_handle(), op, timeout);
    }
#endif

    /**
     * Resumes execution of a direct coroutine handle on this io scheduler.
     * @param handle The coroutine handle to resume execution.
     */
    void resume(std::coroutine_handle<> handle) noexcept override
    {
        if (m_opts.execution_strategy == ExecutionStrategy::process_tasks_inline)
        {
            {
                std::scoped_lock lk{m_scheduled_tasks_mutex};
                m_scheduled_tasks.emplace_back(handle);
            }

            bool expected{false};
            if (m_schedule_fd_triggered.compare_exchange_strong(expected,
                                                                true,
                                                                std::memory_order::release,
                                                                std::memory_order::relaxed))
            {
                eventfd_t value{1};
                eventfd_write(m_schedule_fd, value);
            }
        }
        else
        {
            m_thread_pool->resume(handle);
        }
    }

    /**
     * @return The number of tasks waiting in the task queue + the executing tasks.
     */
    auto size() const noexcept -> std::size_t
    {
        if (m_opts.execution_strategy == ExecutionStrategy::process_tasks_inline)
        {
            return m_size.load(std::memory_order::acquire);
        }

        return m_size.load(std::memory_order::acquire) + m_thread_pool->size();
    }

    /**
     * @return True if the task queue is empty and zero tasks are currently executing.
     */
    auto empty() const noexcept -> bool
    {
        return size() == 0;
    }

    /**
     * Starts the shutdown of the io scheduler.  All currently executing and pending tasks will complete
     * prior to shutting down.  This call is blocking and will not return until all tasks complete.
     */
    auto shutdown() noexcept -> void;

    /**
     * Scans for completed coroutines and destroys them freeing up resources.  This is also done on starting
     * new tasks but this allows the user to cleanup resources manually.  One usage might be making sure fds
     * are cleaned up as soon as possible.
     */
    auto garbage_collect() noexcept -> void;

  private:
    /// The configuration options.
    Options m_opts;

    /// The event loop epoll file descriptor.
    fd_t m_epoll_fd{-1};
    /// The event loop fd to trigger a shutdown.
    fd_t m_shutdown_fd{-1};
    /// The event loop timer fd for timed events, e.g. yield_for() or scheduler_after().
    fd_t m_timer_fd{-1};
    /// The schedule file descriptor if the scheduler is in inline processing mode.
    fd_t m_schedule_fd{-1};
    std::atomic<bool> m_schedule_fd_triggered{false};

    /// The number of tasks executing or awaiting events in this io scheduler.
    std::atomic<std::size_t> m_size{0};

    /// The background io worker threads.
    std::thread m_io_thread;
    /// Thread pool for executing tasks when not in inline mode.
    std::unique_ptr<ThreadPool> m_thread_pool{nullptr};

    std::mutex m_timed_events_mutex{};
    /// The map of time point's to poll infos for tasks that are yielding for a period of time
    /// or for tasks that are polling with timeouts.
    timed_events_t m_timed_events{};

    /// Has the IoScheduler been requested to shut down?
    std::atomic<bool> m_shutdown_requested{false};

    std::atomic<bool> m_io_processing{false};
    auto process_events_manual(std::chrono::milliseconds timeout) -> void;
    auto process_events_dedicated_thread() -> void;
    auto process_events_execute(std::chrono::milliseconds timeout) -> void;
    static auto event_to_poll_status(uint32_t events) -> PollStatus;

    auto process_scheduled_execute_inline() -> void;
    std::mutex m_scheduled_tasks_mutex{};
    std::vector<std::coroutine_handle<>> m_scheduled_tasks{};

    /// Tasks that have their ownership passed into the scheduler.  This is a bit strange for now
    /// but the concept doesn't pass since IoScheduler isn't fully defined yet.
    /// The type is mrc::coroutines::Task_container<mrc::coroutines::IoScheduler>*
    /// Do not inline any functions that use this in the IoScheduler header, it can cause the linker
    /// to complain about "defined in discarded section" because it gets defined multiple times
    void* m_owned_tasks{nullptr};

    static constexpr const int MShutdownObject{0};
    static constexpr const void* MShutdownPtr = &MShutdownObject;

    static constexpr const int MTimerObject{0};
    static constexpr const void* MTimerPtr = &MTimerObject;

    static constexpr const int MScheduleObject{0};
    static constexpr const void* MSchedulePtr = &MScheduleObject;

    static const constexpr std::chrono::milliseconds MDefaultTimeout{1000};
    static const constexpr std::chrono::milliseconds MNoTimeout{0};
    static const constexpr std::size_t MMaxEvents = 16;
    std::array<struct epoll_event, MMaxEvents> m_events{};
    std::vector<std::coroutine_handle<>> m_handles_to_resume{};

    auto process_event_execute(detail::PollInfo* pi, PollStatus status) -> void;
    auto process_timeout_execute() -> void;

    auto add_timer_token(time_point_t tp, detail::PollInfo& pi) -> timed_events_t::iterator;
    auto remove_timer_token(timed_events_t::iterator pos) -> void;
    auto update_timeout(time_point_t now) -> void;
};

}  // namespace mrc::coroutines

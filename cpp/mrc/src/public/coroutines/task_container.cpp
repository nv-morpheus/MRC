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

#include "mrc/coroutines/task_container.hpp"

#include "mrc/coroutines/scheduler.hpp"

#include <glog/logging.h>

#include <mutex>
#include <thread>
#include <utility>

namespace mrc::coroutines {

TaskContainer::StartOperation::StartOperation(TaskContainer& parent, Task<void>&& task, GarbageCollectPolicy cleanup) :
  m_parent(parent),
  m_task(std::move(task)),
  m_cleanup(cleanup)
{}

constexpr auto TaskContainer::StartOperation::await_ready() noexcept -> bool
{
    return false;
}

auto TaskContainer::StartOperation::await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept
{
    m_awaiting_coroutine = awaiting_coroutine;
    return m_parent.schedule_start_operation(this);
}

constexpr auto TaskContainer::StartOperation::await_resume() noexcept -> void {}

TaskContainer::TaskContainer(std::shared_ptr<Scheduler> e, std::size_t concurrency) :
  m_scheduler(std::move(e)),
  m_concurrency(concurrency)
{
    if (m_scheduler == nullptr)
    {
        throw std::runtime_error{"task_container cannot have a nullptr executor"};
    }
}

TaskContainer::~TaskContainer()
{
    // This will hang the current thread.. but if tasks are not complete thats also pretty bad.
    while (!empty())
    {
        garbage_collect();
    }
}

TaskContainer::StartOperation TaskContainer::start(Task<void>&& task, GarbageCollectPolicy cleanup)
{
    return StartOperation{*this, std::move(task), cleanup};
}

auto TaskContainer::garbage_collect() -> std::size_t  // __attribute__((used))
{
    std::lock_guard lk{m_mutex};
    return gc_internal();
}

auto TaskContainer::delete_task_size() const -> std::size_t
{
    std::atomic_thread_fence(std::memory_order::acquire);
    return m_tasks_to_delete.size();
}

auto TaskContainer::delete_tasks_empty() const -> bool
{
    std::atomic_thread_fence(std::memory_order::acquire);
    return m_tasks_to_delete.empty();
}

auto TaskContainer::size() const -> std::size_t
{
    return m_size.load(std::memory_order::relaxed);
}

auto TaskContainer::empty() const -> bool
{
    return size() == 0;
}

auto TaskContainer::garbage_collect_and_yield_until_empty() -> Task<void>
{
    while (!empty())
    {
        garbage_collect();
        co_await m_scheduler->yield();
    }
}

auto TaskContainer::gc_internal() -> std::size_t
{
    std::size_t deleted{0};
    if (!m_tasks_to_delete.empty())
    {
        for (const auto& pos : m_tasks_to_delete)
        {
            if (pos->has_value())
            {
                pos->value().destroy();
            }
            m_tasks.erase(pos);
        }
        deleted = m_tasks_to_delete.size();
        m_tasks_to_delete.clear();
    }
    return deleted;
}

auto TaskContainer::make_cleanup_task(Task<void> user_task, task_position_t pos) -> Task<void>
{
    // Immediately move the task onto the executor.
    co_await m_scheduler->schedule();

    try
    {
        // Await the users task to complete.
        co_await user_task;
    } catch (const std::exception& e)
    {
        // what would be a good way to report this to the user...?  Catching here is required
        // since the co_await will unwrap the unhandled exception on the task.
        // The user's task should ideally be wrapped in a catch all and handle it themselves, but
        // that cannot be guaranteed.
        std::cerr << "Task_container user_task had an unhandled exception e.what()= " << e.what() << "\n";
    } catch (...)
    {
        // don't crash if they throw something that isn't derived from std::exception
        std::cerr << "Task_container user_task had unhandle exception, not derived from std::exception.\n";
    }

    std::unique_lock lock{m_mutex};

    m_size.fetch_sub(1, std::memory_order::relaxed);

    // Check for waiting start operations
    if (!m_waiting_start_operations.empty())
    {
        auto* op = m_waiting_start_operations.front();
        m_waiting_start_operations.pop_front();

        // We must run GC before adding this to the tasks to delete, otherwise we will delete this task while it is
        // still running
        if (op->m_cleanup == GarbageCollectPolicy::yes)
        {
            gc_internal();
        }

        // Defer adding to the tasks to delete until after GC is run
        m_tasks_to_delete.push_back(pos);

        // Call do_start and immediately resume the coroutine.
        auto waiting_coro = this->do_start(std::move(lock), op);

        VLOG(10) << "Resuming handle: " << waiting_coro.address();
        waiting_coro.resume();
    }
    else
    {
        // No waiting start operations, schedule this to be deleted
        m_tasks_to_delete.push_back(pos);
    }

    co_return;
}

auto TaskContainer::do_start(std::unique_lock<std::mutex>&& lock, StartOperation* op) -> std::coroutine_handle<>
{
    // Take ownership of the lock object
    auto local_lock = std::move(lock);

    m_size.fetch_add(1, std::memory_order::relaxed);

    // Store the task inside a cleanup task for self deletion.
    auto pos = m_tasks.emplace(m_tasks.end(), std::nullopt);

    auto task = make_cleanup_task(std::move(op->m_task), pos);

    *pos = std::move(task);

    // Start executing from the cleanup task to schedule the user's task onto the thread pool.
    pos->value().resume();

    return std::move(op->m_awaiting_coroutine);
}

std::coroutine_handle<> TaskContainer::schedule_start_operation(StartOperation* op)
{
    std::unique_lock lock(m_mutex);

    if (m_size < m_concurrency)
    {
        // If op requested a GC before starting, do that now while we have the Mutex
        if (op->m_cleanup == GarbageCollectPolicy::yes)
        {
            gc_internal();
        }

        return this->do_start(std::move(lock), op);
    }

    m_waiting_start_operations.push_back(op);

    return std::noop_coroutine();
}

}  // namespace mrc::coroutines

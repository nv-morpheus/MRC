/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <coroutine>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace mrc::coroutines {

TaskContainer::TaskContainer(std::shared_ptr<Scheduler> e, std::size_t max_concurrent_tasks) :
  m_scheduler_lifetime(std::move(e)),
  m_scheduler(m_scheduler_lifetime.get()),
  m_max_concurrent_tasks(max_concurrent_tasks)
{
    if (m_scheduler_lifetime == nullptr)
    {
        throw std::runtime_error{"TaskContainer cannot have a nullptr executor"};
    }
}

TaskContainer::~TaskContainer()
{
    // This will hang the current thread.. but if tasks are not complete thats also pretty bad.
    while (not empty())
    {
        garbage_collect();
    }
}

auto TaskContainer::start(Task<void>&& user_task, GarbageCollectPolicy cleanup) -> void
{
    auto lock = std::unique_lock(m_mutex);

    m_size += 1;

    if (cleanup == GarbageCollectPolicy::yes)
    {
        gc_internal();
    }

    // Store the task inside a cleanup task for self deletion.
    auto pos  = m_tasks.emplace(m_tasks.end(), std::nullopt);
    auto task = make_cleanup_task(std::move(user_task), pos);
    *pos      = std::move(task);
    m_next_tasks.push(pos);

    auto current_task_count = m_size - m_next_tasks.size();

    if (m_max_concurrent_tasks == 0 or current_task_count < m_max_concurrent_tasks)
    {
        try_start_next_task(std::move(lock));
    }
}

auto TaskContainer::garbage_collect() -> std::size_t
{
    auto lock = std::scoped_lock(m_mutex);
    return gc_internal();
}

auto TaskContainer::size() -> std::size_t
{
    auto lock = std::scoped_lock(m_mutex);
    return m_size;
}

auto TaskContainer::empty() -> bool
{
    return size() == 0;
}

auto TaskContainer::capacity() -> std::size_t
{
    auto lock = std::scoped_lock(m_mutex);
    return m_tasks.size();
}

auto TaskContainer::garbage_collect_and_yield_until_empty() -> Task<void>
{
    while (not empty())
    {
        garbage_collect();
        co_await m_scheduler->yield();
    }
}

TaskContainer::TaskContainer(Scheduler& e) : m_scheduler(&e) {}
auto TaskContainer::gc_internal() -> std::size_t
{
    if (m_tasks_to_delete.empty())
    {
        return 0;
    }

    std::size_t delete_count = m_tasks_to_delete.size();

    for (const auto& pos : m_tasks_to_delete)
    {
        // Destroy the cleanup task and the user task.
        if (pos->has_value())
        {
            pos->value().destroy();
        }

        m_tasks.erase(pos);
    }

    m_tasks_to_delete.clear();

    return delete_count;
}

void TaskContainer::try_start_next_task(std::unique_lock<std::mutex> lock)
{
    if (m_next_tasks.empty())
    {
        // no tasks to process
        return;
    }

    auto pos = m_next_tasks.front();
    m_next_tasks.pop();

    // release the lock before starting the task
    lock.unlock();

    pos->value().resume();
}

auto TaskContainer::make_cleanup_task(Task<void> user_task, task_position_t pos) -> Task<void>
{
    // Immediately move the task onto the executor.
    co_await m_scheduler->yield();

    try
    {
        // Await the users task to complete.
        co_await user_task;
    } catch (const std::exception& e)
    {
        // TODO(MDD): what would be a good way to report this to the user...?  Catching here is required
        // since the co_await will unwrap the unhandled exception on the task.
        // The user's task should ideally be wrapped in a catch all and handle it themselves, but
        // that cannot be guaranteed.
        LOG(ERROR) << "coro::task_container user_task had an unhandled exception e.what()= " << e.what() << "\n";
    } catch (...)
    {
        // don't crash if they throw something that isn't derived from std::exception
        LOG(ERROR) << "coro::task_container user_task had unhandle exception, not derived from std::exception.\n";
    }

    auto lock = std::unique_lock(m_mutex);
    m_tasks_to_delete.push_back(pos);
    // This has to be done within scope lock to make sure this coroutine task completes before the
    // task container object destructs -- if it was waiting on .empty() to become true.
    m_size -= 1;

    try_start_next_task(std::move(lock));

    co_return;
}

}  // namespace mrc::coroutines

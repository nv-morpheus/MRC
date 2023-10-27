/**
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

#include "mrc/coroutines/scheduler.hpp"

#include <glog/logging.h>

#include <ostream>
#include <thread>

namespace mrc::coroutines {

thread_local Scheduler* Scheduler::m_thread_local_scheduler{nullptr};
thread_local std::size_t Scheduler::m_thread_id{0};

Scheduler::Operation::Operation(Scheduler& scheduler) : m_scheduler(scheduler) {}

std::coroutine_handle<> Scheduler::Operation::await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept
{
    m_awaiting_coroutine = awaiting_coroutine;
    return m_scheduler.schedule_operation(this);
}

Scheduler::Scheduler() = default;

auto Scheduler::schedule() -> Operation
{
    return Operation{*this};
}

auto Scheduler::yield() -> Operation
{
    return schedule();
}

auto Scheduler::from_current_thread() noexcept -> Scheduler*
{
    return m_thread_local_scheduler;
}

auto Scheduler::get_thread_id() noexcept -> std::size_t
{
    if (m_thread_local_scheduler == nullptr)
    {
        return std::hash<std::thread::id>()(std::this_thread::get_id());
    }
    return m_thread_id;
}

auto Scheduler::on_thread_start(std::size_t thread_id) -> void
{
    DVLOG(10) << "scheduler: " << description() << " initializing";
    m_thread_id              = thread_id;
    m_thread_local_scheduler = this;
}

}  // namespace mrc::coroutines

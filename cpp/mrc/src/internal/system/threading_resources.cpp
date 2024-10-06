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

#include "internal/system/threading_resources.hpp"

#include "internal/system/fiber_manager.hpp"

#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>

#include <vector>

namespace mrc::system {

ThreadingResources::ThreadingResources(SystemProvider system) :
  SystemProvider(system),
  m_thread_resources(std::make_shared<ThreadResources>(*this)),
  m_fiber_manager(*this)
{
    // Register any initializers and finalizers set on the system object
    for (const auto& f : this->system().thread_initializers())
    {
        this->register_thread_local_initializer(this->system().topology().cpu_set(), f);
    }

    for (const auto& f : this->system().thread_finalizers())
    {
        this->register_thread_local_finalizer(this->system().topology().cpu_set(), f);
    }
}

FiberTaskQueue& ThreadingResources::get_task_queue(std::uint32_t cpu_id) const
{
    return m_fiber_manager.task_queue(cpu_id);
}

FiberPool ThreadingResources::make_fiber_pool(const CpuSet& cpu_set) const
{
    return m_fiber_manager.make_pool(cpu_set);
}

void ThreadingResources::register_thread_local_initializer(const CpuSet& cpu_set, std::function<void()> initializer)
{
    CHECK(initializer);
    CHECK_GE(cpu_set.weight(), 0);
    CHECK(system().topology().contains(cpu_set));
    m_thread_resources->register_initializer(cpu_set, initializer);
    auto futures = m_fiber_manager.enqueue_fiber_on_cpuset(cpu_set, [initializer](std::uint32_t cpu_id) {
        initializer();
    });
    for (auto& f : futures)
    {
        f.get();
    }
}

void ThreadingResources::register_thread_local_finalizer(const CpuSet& cpu_set, std::function<void()> finalizer)
{
    CHECK(finalizer);
    CHECK_GE(cpu_set.weight(), 0);
    CHECK(system().topology().contains(cpu_set));
    m_thread_resources->register_finalizer(cpu_set, finalizer);
}

}  // namespace mrc::system

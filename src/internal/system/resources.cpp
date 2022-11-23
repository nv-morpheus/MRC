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

#include "internal/system/resources.hpp"

#include "internal/system/fiber_manager.hpp"
#include "internal/system/fiber_task_queue.hpp"

#include <boost/fiber/future/future.hpp>

#include <map>
#include <vector>

namespace mrc::internal::system {

Resources::Resources(SystemProvider system) :
  SystemProvider(system),
  m_thread_resources(std::make_shared<ThreadResources>(*this)),
  m_fiber_manager(*this)
{}

FiberTaskQueue& Resources::get_task_queue(std::uint32_t cpu_id) const
{
    return m_fiber_manager.task_queue(cpu_id);
}

FiberPool Resources::make_fiber_pool(const CpuSet& cpu_set) const
{
    return m_fiber_manager.make_pool(cpu_set);
}

void Resources::register_thread_local_initializer(const CpuSet& cpu_set, std::function<void()> initializer)
{
    CHECK(initializer);
    CHECK_GE(cpu_set.weight(), 0);
    CHECK(system().topology().contains(cpu_set));
    m_thread_resources->register_initializer(cpu_set, initializer);
    auto futures =
        m_fiber_manager.enqueue_fiber_on_cpuset(cpu_set, [initializer](std::uint32_t cpu_id) { initializer(); });
    for (auto& f : futures)
    {
        f.get();
    }
}

void Resources::register_thread_local_finalizer(const CpuSet& cpu_set, std::function<void()> finalizer)
{
    CHECK(finalizer);
    CHECK_GE(cpu_set.weight(), 0);
    CHECK(system().topology().contains(cpu_set));
    m_thread_resources->register_finalizer(cpu_set, finalizer);
}

std::unique_ptr<Resources> Resources::unwrap(IResources& resources)
{
    return std::move(resources.m_impl);
}
}  // namespace mrc::internal::system

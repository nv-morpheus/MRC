/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/system/fiber_manager.hpp"

#include "internal/system/fiber_pool.hpp"
#include "internal/system/system.hpp"
#include "internal/system/topology.hpp"

#include <srf/core/bitmap.hpp>
#include <srf/exceptions/runtime_error.hpp>
#include <srf/options/fiber_pool.hpp>
#include <srf/options/options.hpp>

#include <memory>

namespace srf::internal::system {

FiberManager::FiberManager(const System& system) : m_cpu_set(system.topology().cpu_set())
{
    auto cpu_count = system.topology().cpu_set().weight();

    VLOG(1) << "creating fiber task queues on " << cpu_count << " threads";
    VLOG(1) << "thread_binding : " << (system.options().fiber_pool().enable_thread_binding() ? " TRUE" : "FALSE");
    VLOG(1) << "memory_binding : " << (system.options().fiber_pool().enable_memory_binding() ? " TRUE" : "FALSE");

    system.topology().cpu_set().for_each_bit([&](std::int32_t idx, std::int32_t cpu_id) {
        DVLOG(10) << "initializing fiber queue " << idx << " of " << cpu_count << " on cpu_id " << cpu_id;
        m_queues[cpu_id] = std::make_shared<FiberTaskQueue>(system, cpu_id);
    });
}

FiberManager::~FiberManager()
{
    stop();
    join();
}

std::shared_ptr<FiberPool> FiberManager::make_pool(CpuSet cpu_set)
{
    // valididate that cpu_set is a subset of topology->cpu_set()
    if (!m_cpu_set.contains(cpu_set))
    {
        throw exceptions::SrfRuntimeError("cpu_set must be a subset of the initial topology to create a fiber pool");
    }
    auto cpus = cpu_set.vec();
    std::vector<std::shared_ptr<FiberTaskQueue>> queues;
    for (auto& cpu : cpus)
    {
        queues.push_back(m_queues.at(cpu));
    }
    return std::make_shared<FiberPool>(std::move(cpu_set), std::move(queues));
}

void FiberManager::stop()
{
    for (auto& [cpu_id, queue] : m_queues)
    {
        DVLOG(20) << "shutting down fiber running on logical cpu " << cpu_id;
        queue->shutdown();
    }
}

void FiberManager::join()
{
    for (auto& [cpu_id, queue] : m_queues)
    {
        queue.reset();
    }
}

std::shared_ptr<FiberTaskQueue> FiberManager::task_queue(std::uint32_t cpu_id)
{
    auto search = m_queues.find(cpu_id);
    CHECK(search != m_queues.end()) << "unable to find cpu_id " << cpu_id << " in set of task queue";
    return search->second;
}

}  // namespace srf::internal::system

/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/runnable/fiber_engines.hpp"

#include "internal/runnable/fiber_engine.hpp"

#include "mrc/core/fiber_meta_data.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/types.hpp"

#include <glog/logging.h>

#include <memory>
#include <ostream>
#include <string>
#include <utility>

namespace mrc::internal::runnable {

FiberEngines::FiberEngines(system::FiberPool& pool, int priority) :
  FiberEngines(::mrc::runnable::LaunchOptions("custom_options", pool.thread_count()), pool, priority)
{}

FiberEngines::FiberEngines(mrc::runnable::LaunchOptions launch_options, system::FiberPool& pool, int priority) :
  Engines(std::move(launch_options)),
  m_meta{priority}
{
    for (int i = 0; i < pool.thread_count(); i++)
    {
        m_task_queues.emplace_back(pool.task_queue(i));
    }

    initialize_launchers();
}

FiberEngines::FiberEngines(mrc::runnable::LaunchOptions launch_options,
                           system::FiberPool& pool,
                           const FiberMetaData& meta) :
  Engines(std::move(launch_options)),
  m_meta(meta)
{
    for (int i = 0; i < pool.thread_count(); i++)
    {
        m_task_queues.emplace_back(pool.task_queue(i));
    }
    initialize_launchers();
}
FiberEngines::FiberEngines(mrc::runnable::LaunchOptions launch_options,
                           std::vector<std::reference_wrapper<core::FiberTaskQueue>>&& task_queues,
                           int priority) :
  Engines(std::move(launch_options)),
  m_task_queues(std::move(task_queues)),
  m_meta{priority}
{
    initialize_launchers();
}
void FiberEngines::initialize_launchers()
{
    CHECK_EQ(launch_options().pe_count, m_task_queues.size())
        << "mismatched fiber pool task queue size with respect to pe_count";

    for (auto& task_queue : m_task_queues)
    {
        for (int j = 0; j < launch_options().engines_per_pe; ++j)
        {
            Engines::add_launcher(std::make_shared<FiberEngine>(task_queue, m_meta));
        }
    }
}

runnable::EngineType FiberEngines::engine_type() const
{
    return EngineType::Fiber;
}
}  // namespace mrc::internal::runnable

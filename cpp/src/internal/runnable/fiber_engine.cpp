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

#include "internal/runnable/fiber_engine.hpp"

#include "mrc/core/fiber_meta_data.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/runnable/types.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>

#include <type_traits>
#include <utility>

namespace mrc::internal::runnable {

FiberEngine::FiberEngine(core::FiberTaskQueue& task_queue, int priority) : m_task_queue(task_queue), m_meta{priority} {}

FiberEngine::FiberEngine(core::FiberTaskQueue& task_queue, const FiberMetaData& meta) :
  m_task_queue(task_queue),
  m_meta(meta)
{}

Future<void> FiberEngine::do_launch_task(std::function<void()> task)
{
    return m_task_queue.enqueue(m_meta, std::move(task));
}

runnable::EngineType FiberEngine::engine_type() const
{
    return EngineType::Fiber;
}
}  // namespace mrc::internal::runnable

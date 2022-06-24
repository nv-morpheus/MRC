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

#include "srf/core/fiber_pool.hpp"

// Non-main includes
#include "srf/core/bitmap.hpp"

namespace srf::core {

RoundRobinFiberPool::RoundRobinFiberPool(std::shared_ptr<FiberPool> fiber_pool) :
  m_queues(fiber_pool),
  m_provider(fiber_pool->cpu_set())
{}

// std::shared_ptr<core::FiberTaskQueue> RoundRobinFiberPool::next_task_queue()
// {
//     auto index = m_provider.next_index();
//     return m_queues->task_queue_shared(index);
// }

void RoundRobinFiberPool::reset()
{
    m_provider.reset();
}

std::size_t RoundRobinFiberPool::thread_count() const
{
    return m_queues->thread_count();
}

FiberPool& RoundRobinFiberPool::pool()
{
    return *m_queues;
}
}  // namespace srf::core

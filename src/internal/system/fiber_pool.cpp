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

#include "internal/system/fiber_pool.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/core/task_queue.hpp"

#include <ext/alloc_traits.h>
#include <glog/logging.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace mrc::internal::system {

FiberPool::FiberPool(CpuSet cpu_set, std::vector<std::reference_wrapper<FiberTaskQueue>>&& queues) :
  m_cpu_set(std::move(cpu_set)),
  m_queues(std::move(queues))
{}

const CpuSet& FiberPool::cpu_set() const
{
    return m_cpu_set;
}

std::size_t FiberPool::thread_count() const
{
    return m_queues.size();
}

core::FiberTaskQueue& FiberPool::task_queue(const std::size_t& index)
{
    CHECK_LT(index, m_queues.size());
    return m_queues.at(index);
}

}  // namespace mrc::internal::system

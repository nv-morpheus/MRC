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

#include "internal/system/fiber_task_queue.hpp"

#include "internal/system/fiber_priority_scheduler.hpp"
#include "internal/system/resources.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/core/fiber_meta_data.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/channel_op_status.hpp>
#include <boost/fiber/context.hpp>
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/future/future.hpp>
#include <boost/fiber/future/packaged_task.hpp>
#include <boost/fiber/operations.hpp>
#include <glog/logging.h>

#include <ostream>
#include <string>
#include <thread>
#include <utility>

namespace mrc::internal::system {

FiberTaskQueue::FiberTaskQueue(const Resources& resources, CpuSet cpu_affinity, std::size_t channel_size) :
  m_queue(channel_size),
  m_cpu_affinity(std::move(cpu_affinity)),
  m_thread(resources.make_thread("fiberq", m_cpu_affinity, [this] { main(); }))
{
    DVLOG(10) << "awaiting fiber task queue worker thread running on cpus " << m_cpu_affinity;
    enqueue([] {}).get();
    DVLOG(10) << *this << ": ready";
}

FiberTaskQueue::~FiberTaskQueue()
{
    shutdown();
}

const CpuSet& FiberTaskQueue::affinity() const
{
    return m_cpu_affinity;
}

boost::fibers::buffered_channel<core::FiberTaskQueue::task_pkg_t>& FiberTaskQueue::task_queue()
{
    return m_queue;
}

void FiberTaskQueue::main()
{
    // enable priority scheduler
    boost::fibers::use_scheduling_algorithm<FiberPriorityScheduler>();

    task_pkg_t task_pkg;
    while (true)
    {
        auto rc = m_queue.pop(task_pkg);
        if (rc == boost::fibers::channel_op_status::closed)
        {
            break;
        }
        launch(std::move(task_pkg));
    }

    if (detached() != 0U)
    {
        VLOG(10) << *this << ": waiting on detached fibers";
    }

    while (detached() != 0U)
    {
        boost::this_fiber::yield();
    }

    VLOG(10) << *this << ": completed";
}

void FiberTaskQueue::shutdown()
{
    m_queue.close();
}

void FiberTaskQueue::launch(task_pkg_t&& pkg) const
{
    // default is a post, not a dispatch, so the task is only enqueued with the fiber scheduler
    boost::fibers::fiber fiber(std::move(pkg.first));
    auto& props(fiber.properties<FiberPriorityProps>());
    props.set_priority(pkg.second.priority);
    DVLOG(10) << *this << ": created fiber " << fiber.get_id() << " with priority " << pkg.second.priority;
    fiber.detach();
}

std::ostream& operator<<(std::ostream& os, const FiberTaskQueue& ftq)
{
    if (ftq.affinity().weight() == 1)
    {
        os << "[fiber_task_queue: cpu_id: " << ftq.affinity().first() << "; tid: " << ftq.m_thread.thread().get_id()
           << "]";
    }
    else
    {
        os << "[fiber_task_queue: on cpus: " << ftq.affinity().str() << "; tid: " << ftq.m_thread.thread().get_id()
           << "]";
    }
    return os;
}

std::thread::id FiberTaskQueue::thread_id() const
{
    return m_thread.thread().get_id();
}
bool FiberTaskQueue::caller_on_same_thread() const
{
    return std::this_thread::get_id() == m_thread.thread().get_id();
}
}  // namespace mrc::internal::system

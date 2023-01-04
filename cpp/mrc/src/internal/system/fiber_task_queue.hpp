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

#pragma once

#include "internal/system/thread.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/utils/macros.hpp"

#include <boost/fiber/buffered_channel.hpp>

#include <cstddef>
#include <iosfwd>
#include <thread>

namespace mrc::internal::system {

class Resources;

class FiberTaskQueue final : public core::FiberTaskQueue
{
  public:
    FiberTaskQueue(const Resources& resources, CpuSet cpu_affinity, std::size_t channel_size = 64);
    ~FiberTaskQueue() final;

    DELETE_COPYABILITY(FiberTaskQueue);
    DELETE_MOVEABILITY(FiberTaskQueue);

    const CpuSet& affinity() const final;

    std::thread::id thread_id() const final;
    bool caller_on_same_thread() const final;

    void shutdown();

    friend std::ostream& operator<<(std::ostream& os, const FiberTaskQueue& ftq);

  private:
    void main();
    void launch(task_pkg_t&& pkg) const;

    boost::fibers::buffered_channel<task_pkg_t>& task_queue() final;

    boost::fibers::buffered_channel<task_pkg_t> m_queue;
    CpuSet m_cpu_affinity;
    Thread m_thread;
};

}  // namespace mrc::internal::system

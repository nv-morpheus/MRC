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

#include "internal/system/thread_pool.hpp"

#include "internal/system/resources.hpp"

#include "mrc/core/bitmap.hpp"

#include <cstdint>
#include <thread>

namespace mrc::internal::system {

ThreadPool::ThreadPool(const system::Resources& resources, CpuSet cpuset, std::size_t channel_size) :
  m_cpuset(std::move(cpuset)),
  m_channel(channel_size)
{
    m_cpuset.for_each_bit([this, &resources](std::uint32_t idx, std::uint32_t bit) {
        m_threads.emplace_back(resources.make_thread("thread_pool", CpuSet(bit), [this] {
            boost::fibers::channel_op_status status;
            do
            {
                std::packaged_task<void()> task;
                status = m_channel.pop(task);
                if (status == boost::fibers::channel_op_status::success)
                {
                    DVLOG(10) << "[thread_pool; tid=" << std::this_thread::get_id() << "]: executing enqueued task";
                    task();
                    DVLOG(10) << "[thread_pool; tid=" << std::this_thread::get_id() << "]: task complete";
                }
            } while (status == boost::fibers::channel_op_status::success);

            DVLOG(10) << "[thread_pool; tid=" << std::this_thread::get_id() << "]: exiting primary run loop";
        }));
    });
}

ThreadPool::~ThreadPool()
{
    if (!m_channel.is_closed())
    {
        this->shutdown();
    }
    DVLOG(10) << "[thread_pool]: joining threads";
}

void ThreadPool::shutdown()
{
    DVLOG(10) << "[thread_pool]: shutdown requested; closing channel";
    m_channel.close();
}

}  // namespace mrc::internal::system

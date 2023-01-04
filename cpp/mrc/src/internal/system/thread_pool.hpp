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

#include "internal/system/resources.hpp"
#include "internal/system/thread.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/exceptions/runtime_error.hpp"

#include <boost/fiber/buffered_channel.hpp>
#include <boost/fiber/channel_op_status.hpp>
#include <boost/fiber/future/future.hpp>
#include <boost/fiber/future/packaged_task.hpp>
#include <glog/logging.h>

#include <cstddef>
#include <functional>
#include <future>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace mrc::internal::system {

/**
 * @brief Fiber-friendly ThreadPool
 *
 * Front-end is fiber friendly; backend tasks are run on 1 or more std::threads.
 *
 * Tasks submitted to this ThreadPool are meant to be compute intensive tasks that will not yield the thread execution
 * context. This is in stark contrast to the FiberPool which is designed to run a bunch of asynchronous task which yield
 * the thread while awaiting rescheduling. The primary motivation of the ThreadPool is to offload compute intensive
 * tasks from the fibers driving forward progress in the pipelines event loops.
 *
 * The enqueue method and its return value are fiber friendly, meaning that if the work queue (channel_size) is full,
 * then the enqueue method will yield the fiber. Similarly, the return value of the enqueue method is a
 * boost::fibers::future of the return type of the callable F.
 *
 * The backend consists of a set of std::threads that execute std::packaged_task<void()> which they pull off the fiber
 * channel. Tasks should not independently spawn fibers.
 *
 */
class ThreadPool final
{
  public:
    ThreadPool(const system::Resources&, CpuSet cpuset, std::size_t channel_size = 128);
    ~ThreadPool();

    template <class F, class... ArgsT>
    auto enqueue(F&& f, ArgsT&&... args) -> boost::fibers::future<typename std::result_of<F(ArgsT...)>::type>
    {
        using return_type_t = typename std::result_of<F(ArgsT...)>::type;

        boost::fibers::packaged_task<return_type_t()> task(std::bind(std::forward<F>(f), std::forward<ArgsT>(args)...));
        boost::fibers::future<return_type_t> future = task.get_future();

        std::packaged_task<void()> wrapped_task([this, t = std::move(task)]() mutable { t(); });
        auto status = m_channel.push(std::move(wrapped_task));

        if (status == boost::fibers::channel_op_status::closed)
        {
            LOG(ERROR) << "failed to enqueue work to ThreadPool; ThreadPool is shutting down";
            throw exceptions::MrcRuntimeError("failed to enqueue work to ThreadPool; ThreadPool is shutting down");
        }

        return std::move(future);
    }

    void shutdown();

  private:
    const CpuSet m_cpuset;
    boost::fibers::buffered_channel<std::packaged_task<void()>> m_channel;
    std::vector<system::Thread> m_threads;
};

}  // namespace mrc::internal::system

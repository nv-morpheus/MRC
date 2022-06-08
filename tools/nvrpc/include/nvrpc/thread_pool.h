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

//
// Original Source: https://github.com/progschj/ThreadPool
//
// Original License:
//
// Copyright (c) 2012 Jakob Progsch, Václav Zeman
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
//   1. The origin of this software must not be misrepresented; you must not
//   claim that you wrote the original software. If you use this software
//   in a product, an acknowledgment in the product documentation would be
//   appreciated but is not required.
//
//   2. Altered source versions must be plainly marked as such, and must not be
//   misrepresented as being the original software.
//
//   3. This notice may not be removed or altered from any source
//   distribution.
//
// Modifications:
//   * Header-only file was split into .h/.cc files
//   * Added an extra safety check (lines 30-31) in the construction (.cc file).
//   * Added CPU affinity options to the constructor
//   * Added Size() method to get thread count
//   * Implemented transwarp::executor protocol
//
#pragma once

#include <functional>
#include <future>
#include <queue>
#include <vector>

#include <glog/logging.h>

namespace nvrpc {

template <typename MutexType, typename ConditionType>
class BaseThreadPool;

/**
 * @brief Manages a Pool of Threads that consume a shared work Queue
 *
 * BaseThreadPool is the primary resoruce class for handling threads used throughout
 * the YAIS examples and tests.  The library is entirely a BYO-resources;
 * however, this implemenation is provided as a convenience class.  Many thanks
 * to the original authors for a beautifully designed class.
 */
template <typename MutexType, typename ConditionType>
class BaseThreadPool
{
  public:
    /**
     * @brief Construct a new Thread Pool
     * @param nThreads Number of Worker Threads
     */
    BaseThreadPool(size_t nThreads);

    virtual ~BaseThreadPool();

    BaseThreadPool(const BaseThreadPool&) = delete;
    BaseThreadPool& operator=(const BaseThreadPool&) = delete;

    BaseThreadPool(BaseThreadPool&&) = delete;
    BaseThreadPool& operator=(BaseThreadPool&&) = delete;

    /**
     * @brief Enqueue Work to the BaseThreadPool by passing a Lambda Function
     *
     * Variadic template allows for an arbituary number of arguments to be passed
     * the captured lambda function.  Captures are still allowed and used
     * throughout the examples.
     *
     * The queue can grow larger than the number of threads.  A single worker
     * thread executues pulls a lambda function off the queue and executes it to
     * completion.  These are synchronous executions in an async messaging
     * library.  These synchronous pools can be swapped for truely async workers
     * using libevent or asio.  Happy to accept PRs to improve the async
     * abilities.
     *
     * @tparam F
     * @tparam Args
     * @param f
     * @param args
     * @return std::future<typename std::result_of<F(Args...)>::type>
     */
    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

    void enqueue(std::function<void()> task);

    /**
     * @brief Number of Threads in the Pool
     */
    int size() const;

  private:
    void create_thread();

    std::vector<std::thread> m_workers;
    std::queue<std::packaged_task<void()>> m_tasks;

    // synchronization
    MutexType m_QueueMutex;
    ConditionType m_Condition;
    bool stop;
};

// add new work item to the pool
template <typename MutexType, typename ConditionType>
template <class F, class... Args>
auto BaseThreadPool<MutexType, ConditionType>::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    std::packaged_task<return_type()> task(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task.get_future();
    {
        std::lock_guard<MutexType> lock(m_QueueMutex);

        // don't allow enqueueing after stopping the pool
        if (stop)
            throw std::runtime_error("enqueue on stopped BaseThreadPool");

        m_tasks.emplace(std::move(task));
    }
    m_Condition.notify_one();
    return res;
}

template <typename MutexType, typename ConditionType>
BaseThreadPool<MutexType, ConditionType>::BaseThreadPool(size_t thread_count) : stop(false)
{
    for (size_t i = 0; i < thread_count; ++i)
    {
        create_thread();
    }
}

template <typename MutexType, typename ConditionType>
void BaseThreadPool<MutexType, ConditionType>::create_thread()
{
    m_workers.emplace_back([this]() {
        // affinity::this_thread::set_affinity(affinity_mask);
        for (;;)
        {
            std::packaged_task<void()> task;
            {
                std::unique_lock<MutexType> lock(this->m_QueueMutex);
                this->m_Condition.wait(lock, [this]() { return this->stop || !this->m_tasks.empty(); });
                if (this->stop && this->m_tasks.empty())
                    return;
                task = move(this->m_tasks.front());
                this->m_tasks.pop();
            }
            task();
        }
    });
}

// the destructor joins all threads
template <typename MutexType, typename ConditionType>
BaseThreadPool<MutexType, ConditionType>::~BaseThreadPool()
{
    {
        std::lock_guard<MutexType> lock(m_QueueMutex);
        stop = true;
    }
    m_Condition.notify_all();

    for (std::thread& worker : m_workers)
    {
        worker.join();
    }
}

template <typename MutexType, typename ConditionType>
int BaseThreadPool<MutexType, ConditionType>::size() const
{
    return m_workers.size();
}

using ThreadPool = BaseThreadPool<std::mutex, std::condition_variable>;

}  // namespace nvrpc

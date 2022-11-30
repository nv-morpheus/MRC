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

#include "mrc/core/thread_barrier.hpp"
#include "mrc/forward.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/forward.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>

#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace mrc::runnable {

class ThreadContextResources
{
  public:
    ThreadContextResources() = delete;
    ThreadContextResources(std::size_t size) : m_barrier(size) {}
    virtual ~ThreadContextResources() = default;

    void barrier()
    {
        m_barrier.wait();
    }

    std::mutex& mutex()
    {
        return m_mutex;
    }

  private:
    thread_barrier m_barrier;
    std::mutex m_mutex;
};

/**
 * @brief Final specialization of a ContextT for use with a FiberRunnable
 *
 * @tparam ContextT
 */
template <typename ContextT = Context>
class ThreadContext final : public ContextT
{
  public:
    using resource_t = ThreadContextResources;

    template <typename... ArgsT>
    explicit ThreadContext(std::shared_ptr<ThreadContextResources> resources, ArgsT&&... args) :
      ContextT(std::forward<ArgsT>(args)...),
      m_resources(std::move(resources)),
      m_lock(m_resources->mutex())
    {
        // the mutex is locked on the construction of m_lock
        m_lock.unlock();
    }

    ThreadContext()           = delete;
    ~ThreadContext() override = default;

  private:
    /**
     * @brief called on the fiber before main is run
     */
    void init_info(std::stringstream& ss) final
    {
        ContextT::init_info(ss);
        ss << "; tid: " << std::this_thread::get_id();
    }

    void do_lock() final
    {
        m_lock.lock();
    }

    void do_unlock() final
    {
        m_lock.unlock();
    }

    void do_barrier() final
    {
        m_resources->barrier();
    }

    void do_yield() final
    {
        std::this_thread::yield();
    }

    EngineType do_execution_context() const final
    {
        return EngineType::Thread;
    }

    std::shared_ptr<ThreadContextResources> m_resources;
    std::unique_lock<std::mutex> m_lock;
};

}  // namespace mrc::runnable

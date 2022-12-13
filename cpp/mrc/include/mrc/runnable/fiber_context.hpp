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

#include "mrc/forward.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/forward.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/utils/type_utils.hpp"

#include <boost/fiber/barrier.hpp>
#include <boost/fiber/mutex.hpp>
#include <boost/fiber/operations.hpp>

#include <sstream>  // for stringstream
#include <string>

namespace mrc::runnable {

class FiberContextResources
{
  public:
    FiberContextResources() = delete;
    FiberContextResources(std::size_t size) : m_barrier(size) {}
    virtual ~FiberContextResources() = default;

    void barrier()
    {
        m_barrier.wait();
    }

    boost::fibers::mutex& mutex()
    {
        return m_mutex;
    }

  private:
    boost::fibers::barrier m_barrier;
    boost::fibers::mutex m_mutex;
};

/**
 * @brief Final specialization of a ContextT for use with a FiberRunnable
 *
 * @tparam ContextT
 */
template <typename ContextT = Context>
class FiberContext final : public ContextT
{
  public:
    using resource_t = FiberContextResources;

    template <typename... ArgsT>
    explicit FiberContext(std::shared_ptr<FiberContextResources> fiber_resources, ArgsT&&... args) :
      ContextT(std::forward<ArgsT>(args)...),
      m_fiber_resources(std::move(fiber_resources)),
      m_lock(m_fiber_resources->mutex())
    {
        // the mutex is locked on the construction of m_lock
        m_lock.unlock();
    }

    FiberContext()           = delete;
    ~FiberContext() override = default;

  private:
    /**
     * @brief called on the fiber before main is run
     */
    void init_info(std::stringstream& ss) final
    {
        ContextT::init_info(ss);
        ss << "; tid: " << std::this_thread::get_id() << "; fid: " << boost::this_fiber::get_id();
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
        m_fiber_resources->barrier();
    }

    void do_yield() final
    {
        boost::this_fiber::yield();
    }

    EngineType do_execution_context() const final
    {
        return EngineType::Fiber;
    }

    std::shared_ptr<FiberContextResources> m_fiber_resources;
    std::unique_lock<boost::fibers::mutex> m_lock;
};

}  // namespace mrc::runnable

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

#include "internal/runnable/engine_factory.hpp"

#include "internal/runnable/fiber_engines.hpp"
#include "internal/runnable/thread_engines.hpp"
#include "internal/system/fiber_pool.hpp"
#include "internal/system/resources.hpp"

#include "srf/constants.hpp"
#include "srf/core/bitmap.hpp"
#include "srf/core/task_queue.hpp"
#include "srf/exceptions/runtime_error.hpp"
#include "srf/runnable/engine.hpp"
#include "srf/runnable/engine_factory.hpp"
#include "srf/runnable/launch_options.hpp"
#include "srf/runnable/types.hpp"

#include <glog/logging.h>

#include <cstddef>
#include <functional>
#include <ostream>
#include <utility>
#include <vector>

namespace srf::internal::runnable {

class FiberEngineFactory : public ::srf::runnable::EngineFactory
{
  public:
    /**
     * @brief FiberEngines will be build on N pes/threads with fibers per thread equivalent to engines_per_pe.
     *
     * @param pes - the number of processing elements (threads) on which some number of fibers will be enqueued
     * @param engines_per_pe - number of fibers per thread
     *
     * @return std::shared_ptr<Engines>
     */
    std::shared_ptr<::srf::runnable::Engines> build_engines(const LaunchOptions& launch_options) final
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        return std::make_shared<FiberEngines>(
            launch_options, get_next_n_queues(launch_options.pe_count), SRF_DEFAULT_FIBER_PRIORITY);
    }

    ::srf::runnable::EngineType backend() const final
    {
        return EngineType::Fiber;
    }

  private:
    virtual std::vector<std::reference_wrapper<core::FiberTaskQueue>> get_next_n_queues(std::size_t count) = 0;
    std::mutex m_mutex;
};

/**
 * @brief Used to construct FiberEngines to launch Runnables
 */
class ReusableFiberEngineFactory final : public FiberEngineFactory
{
  public:
    ReusableFiberEngineFactory(const system::Resources& system_resources, const CpuSet& cpu_set) :
      m_pool(system_resources.make_fiber_pool(cpu_set))
    {}
    ~ReusableFiberEngineFactory() final = default;

    std::vector<std::reference_wrapper<core::FiberTaskQueue>> get_next_n_queues(std::size_t count) final
    {
        DCHECK_LE(count, m_pool.thread_count());
        std::vector<std::reference_wrapper<core::FiberTaskQueue>> queues;

        for (int i = 0; i < count; ++i)
        {
            queues.emplace_back(m_pool.task_queue(next()));
        }

        return std::move(queues);
    }

  private:
    std::size_t next()
    {
        auto n = m_offset++;
        if (m_offset == m_pool.thread_count())
        {
            m_offset = 0;
        }
        return n;
    }

    system::FiberPool m_pool;
    std::size_t m_offset{0};
};

/**
 * @brief Used to construct FiberEngines to launch Runnables
 */
class SingleUseFiberEngineFactory final : public FiberEngineFactory
{
  public:
    SingleUseFiberEngineFactory(const system::Resources& system_resources, const CpuSet& cpu_set) :
      m_pool(system_resources.make_fiber_pool(cpu_set))
    {}
    ~SingleUseFiberEngineFactory() final = default;

  protected:
    std::vector<std::reference_wrapper<core::FiberTaskQueue>> get_next_n_queues(std::size_t count) final
    {
        if (m_offset + count > m_pool.thread_count())
        {
            LOG(ERROR) << "more dedicated threads/cores than available";
            throw exceptions::SrfRuntimeError("more dedicated threads/cores than available");
        }

        std::vector<std::reference_wrapper<core::FiberTaskQueue>> queues;

        for (int i = 0; i < count; ++i)
        {
            queues.emplace_back(m_pool.task_queue(m_offset + i));
        }

        return std::move(queues);
    }

  private:
    system::FiberPool m_pool;
    std::size_t m_offset{0};
};

class ThreadEngineFactory : public ::srf::runnable::EngineFactory
{
  public:
    ThreadEngineFactory(const system::Resources& system_resources, CpuSet cpu_set) :
      m_system_resources(system_resources),
      m_cpu_set(std::move(cpu_set))
    {
        CHECK(!m_cpu_set.empty());
    }

    std::shared_ptr<::srf::runnable::Engines> build_engines(const LaunchOptions& launch_options) final
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        auto cpu_set = get_next_n_cpus(launch_options.pe_count);
        return std::make_shared<ThreadEngines>(launch_options, std::move(cpu_set), m_system_resources);
    }

  protected:
    const CpuSet& cpu_set() const
    {
        return m_cpu_set;
    }

    EngineType backend() const final
    {
        return EngineType::Thread;
    }

  private:
    virtual CpuSet get_next_n_cpus(std::size_t count) = 0;

    CpuSet m_cpu_set;
    const system::Resources& m_system_resources;
    std::mutex m_mutex;
};

/**
 * @brief Used to construct ThreadEngines to launch Runnables
 */
class ReusableThreadEngineFactory final : public ThreadEngineFactory
{
  public:
    ReusableThreadEngineFactory(const system::Resources& system_resources, const CpuSet& cpu_set) :
      ThreadEngineFactory(system_resources, cpu_set)
    {}

  protected:
    CpuSet get_next_n_cpus(std::size_t count) final
    {
        CpuSet cpu_set;
        for (int i = 0; i < count; ++i)
        {
            m_prev_cpu_idx = this->cpu_set().next(m_prev_cpu_idx);
            if (m_prev_cpu_idx == -1)
            {
                m_prev_cpu_idx = this->cpu_set().next(m_prev_cpu_idx);
            }
            cpu_set.on(m_prev_cpu_idx);
        }
        return cpu_set;
    }

  private:
    int m_prev_cpu_idx = -1;
};

class SingleUseThreadEngineFactory final : public ThreadEngineFactory
{
  public:
    SingleUseThreadEngineFactory(const system::Resources& system_resources, const CpuSet& cpu_set) :
      ThreadEngineFactory(system_resources, cpu_set)
    {}

  protected:
    CpuSet get_next_n_cpus(std::size_t count) final
    {
        CpuSet cpu_set;
        for (int i = 0; i < count; ++i)
        {
            m_prev_cpu_idx = this->cpu_set().next(m_prev_cpu_idx);
            if (m_prev_cpu_idx == -1)
            {
                LOG(ERROR) << "SingleUse logical cpu ids exhausted";
                throw exceptions::SrfRuntimeError("SingleUse logical cpu ids exhausted");
            }
            cpu_set.on(m_prev_cpu_idx);
        }
        return cpu_set;
    }

  private:
    int m_prev_cpu_idx = -1;
};

std::shared_ptr<::srf::runnable::EngineFactory> make_engine_factory(const system::Resources& system_resources,
                                                                    EngineType engine_type,
                                                                    const CpuSet& cpu_set,
                                                                    bool reusable)
{
    if (engine_type == EngineType::Fiber)
    {
        if (reusable)
        {
            return std::make_shared<ReusableFiberEngineFactory>(system_resources, cpu_set);
        }
        return std::make_shared<SingleUseFiberEngineFactory>(system_resources, cpu_set);
    }

    if (engine_type == EngineType::Thread)
    {
        if (reusable)
        {
            return std::make_shared<ReusableThreadEngineFactory>(system_resources, cpu_set);
        }
        return std::make_shared<SingleUseThreadEngineFactory>(system_resources, cpu_set);
    }

    LOG(FATAL) << "unsupported engine type";
}

}  // namespace srf::internal::runnable

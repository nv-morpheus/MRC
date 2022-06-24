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

#include "internal/system/fiber_manager.hpp"
#include "internal/system/fiber_pool.hpp"
#include "internal/system/fiber_task_queue.hpp"
#include "internal/system/topology.hpp"

#include "srf/core/bitmap.hpp"
#include "srf/options/options.hpp"
#include "srf/types.hpp"
#include "srf/utils/macros.hpp"
#include "srf/utils/thread_local_shared_pointer.hpp"

#include <glog/logging.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <thread>
#include <utility>

namespace srf::internal::system {
class Partitions;
}

namespace srf::internal::system {

class ISystem;

class System final
{
    System(std::shared_ptr<const Options> options);

  public:
    static std::shared_ptr<System> make_system(std::shared_ptr<Options> options);
    static std::shared_ptr<System> unwrap(const ISystem& system);

    ~System();

    DELETE_COPYABILITY(System);
    DELETE_MOVEABILITY(System);

    const Options& options() const;
    const Topology& topology() const;
    const Partitions& partitions() const;

    template <typename CallableT>
    std::thread make_thread(CpuSet cpu_affinity, CallableT&& callable) const;

    template <typename CallableT>
    std::thread make_thread(std::string desc, CpuSet cpu_affinity, CallableT&& callable) const;

    std::shared_ptr<FiberTaskQueue> get_task_queue(std::uint32_t cpu_id) const;
    std::shared_ptr<FiberPool> make_fiber_pool(CpuSet cpu_set) const;

    template <typename ResourceT>
    void register_thread_local_resource(const CpuSet& cpu_set, std::shared_ptr<ResourceT> resource);

    void register_thread_local_initializer(const CpuSet& cpu_set, std::function<void()> initializer);
    void register_thread_local_finalizer(const CpuSet& cpu_set, std::function<void()> initializer);

    CpuSet get_current_thread_affinity() const;

  private:
    void initialize_thread(const std::string& desc, const CpuSet& cpu_affinity) const;
    void finalize_thread(const CpuSet& cpu_affinity) const;

    // default constructed - must be constructed before initializer list
    std::multimap<int, std::function<void()>> m_thread_initializers;
    std::multimap<int, std::function<void()>> m_thread_finalizers;
    mutable std::atomic<bool> m_cap_membind{true};

    // constructor initializer list
    Handle<const Options> m_options;
    Handle<Topology> m_topology;
    Handle<Partitions> m_partitions;
    Handle<FiberManager> m_fiber_manager;  // possibly move to resources
};

template <typename ResourceT>
void System::register_thread_local_resource(const CpuSet& cpu_set, std::shared_ptr<ResourceT> resource)
{
    CHECK(resource);
    CHECK(topology().contains(cpu_set));
    auto pool = make_fiber_pool(cpu_set);
    pool->set_thread_local_resource(resource);
    register_thread_local_initializer([resource] { ::srf::utils::ThreadLocalSharedPointer<ResourceT>::set(resource); });
}

template <typename CallableT>
std::thread System::make_thread(CpuSet cpu_affinity, CallableT&& callable) const
{
    return make_thread("thread", std::move(cpu_affinity), std::move(callable));
}

template <typename CallableT>
std::thread System::make_thread(std::string desc, CpuSet cpu_affinity, CallableT&& callable) const
{
    CHECK(topology().contains(cpu_affinity));
    return std::thread([this, desc, cpu_affinity, thread_task = std::move(callable)]() mutable {
        DVLOG(10) << "tid: " << std::this_thread::get_id() << "; initializing thread";
        initialize_thread(desc, cpu_affinity);
        DVLOG(10) << "tid: " << std::this_thread::get_id() << "; execute thread task";
        thread_task();
        DVLOG(10) << "tid: " << std::this_thread::get_id() << "; finalize thread";
        finalize_thread(cpu_affinity);
        DVLOG(10) << "tid: " << std::this_thread::get_id() << "; completed thread";
    });
}

inline std::shared_ptr<System> make_system(std::shared_ptr<Options> options)
{
    return System::make_system(std::move(options));
}

}  // namespace srf::internal::system

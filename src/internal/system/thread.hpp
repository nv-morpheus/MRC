/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/system/system.hpp"
#include "internal/system/system_provider.hpp"
#include "internal/system/topology.hpp"

#include "srf/core/bitmap.hpp"

#include <glog/logging.h>

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <thread>
#include <utility>

namespace srf::internal::system {

class ThreadResources;

/**
 * @brief Non-detachable Thread object that will join on destruction.
 *
 * To ensure the initializer and finalizer methods used within the threads are available on thread completion, we create
 * this jthread-like class that also associates some set of resources that who lifespan will be maintained.
 */
class Thread
{
  public:
    ~Thread();
    const std::thread& thread() const;

    Thread(Thread&&) noexcept = default;
    Thread& operator=(Thread&&) noexcept = default;

    void join();

  private:
    Thread(std::shared_ptr<const ThreadResources> resources, std::thread&& thread);
    std::shared_ptr<const ThreadResources> m_resources;
    std::thread m_thread;  // use std::jthread in c++20; require std::stop_token in task signature

    friend ThreadResources;
};

/**
 * @brief Resources for all SRF threads
 *
 * The ThreadResources holds the thread initializer and finalizer methods, and is responsible for spawning Threads which
 * capture a shared_ptr<ThreadResources> via shared_from_this();
 */
class ThreadResources final : private SystemProvider, public std::enable_shared_from_this<ThreadResources>
{
  public:
    ThreadResources(const SystemProvider& system) : SystemProvider(system) {}

    void register_initializer(const CpuSet& cpu_set, std::function<void()> initializer);
    void register_finalizer(const CpuSet& cpu_set, std::function<void()> finalizer);

    template <typename CallableT>
    Thread make_thread(std::string desc, CpuSet cpu_affinity, CallableT&& callable) const;

  private:
    void initialize_thread(const std::string& desc, const CpuSet& cpu_affinity) const;
    void finalize_thread(const CpuSet& cpu_affinity) const;

    std::multimap<int, std::function<void()>> m_thread_initializers;
    std::multimap<int, std::function<void()>> m_thread_finalizers;
    mutable std::atomic<bool> m_cap_membind{true};
};

template <typename CallableT>
Thread ThreadResources::make_thread(std::string desc, CpuSet cpu_affinity, CallableT&& callable) const
{
    CHECK(cpu_affinity.weight());
    CHECK(system().topology().contains(cpu_affinity));
    auto thread = std::thread([this, desc, cpu_affinity, thread_task = std::move(callable)]() mutable {
        DVLOG(10) << "tid: " << std::this_thread::get_id() << "; initializing thread";
        initialize_thread(desc, cpu_affinity);
        DVLOG(10) << "tid: " << std::this_thread::get_id() << "; execute thread task";
        thread_task();
        DVLOG(10) << "tid: " << std::this_thread::get_id() << "; finalize thread";
        finalize_thread(cpu_affinity);
        DVLOG(10) << "tid: " << std::this_thread::get_id() << "; completed thread";
    });
    return Thread(shared_from_this(), std::move(thread));
}

}  // namespace srf::internal::system

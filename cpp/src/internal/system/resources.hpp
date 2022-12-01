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

#include "internal/system/fiber_manager.hpp"
#include "internal/system/fiber_pool.hpp"
#include "internal/system/fiber_task_queue.hpp"
#include "internal/system/system.hpp"
#include "internal/system/system_provider.hpp"
#include "internal/system/thread.hpp"
#include "internal/system/topology.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/engine/system/iresources.hpp"
#include "mrc/utils/thread_local_shared_pointer.hpp"

#include <glog/logging.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace mrc::internal::system {

class Resources final : public SystemProvider
{
  public:
    static std::unique_ptr<Resources> unwrap(IResources& resources);
    static std::unique_ptr<Resources> create(const SystemProvider& system);
    static std::unique_ptr<Resources> create(std::shared_ptr<const SystemProvider> system);

    Resources(SystemProvider system);

    template <typename CallableT>
    [[nodiscard]] Thread make_thread(CpuSet cpu_affinity, CallableT&& callable) const;

    template <typename CallableT>
    [[nodiscard]] Thread make_thread(std::string desc, CpuSet cpu_affinity, CallableT&& callable) const;

    FiberPool make_fiber_pool(const CpuSet& cpu_set) const;
    FiberTaskQueue& get_task_queue(std::uint32_t cpu_id) const;

    template <typename ResourceT>
    void register_thread_local_resource(const CpuSet& cpu_set, std::shared_ptr<ResourceT> resource);

    void register_thread_local_initializer(const CpuSet& cpu_set, std::function<void()> initializer);
    void register_thread_local_finalizer(const CpuSet& cpu_set, std::function<void()> finalizer);

  private:
    std::shared_ptr<ThreadResources> m_thread_resources;
    FiberManager m_fiber_manager;
};

template <typename ResourceT>
void Resources::register_thread_local_resource(const CpuSet& cpu_set, std::shared_ptr<ResourceT> resource)
{
    CHECK(resource);
    CHECK(system().topology().contains(cpu_set));
    auto pool = make_fiber_pool(cpu_set);
    pool.set_thread_local_resource(resource);
    register_thread_local_initializer([resource] { ::mrc::utils::ThreadLocalSharedPointer<ResourceT>::set(resource); });
}

template <typename CallableT>
Thread Resources::make_thread(CpuSet cpu_affinity, CallableT&& callable) const
{
    CHECK(m_thread_resources);
    return m_thread_resources->make_thread("thread", std::move(cpu_affinity), std::move(callable));
}

template <typename CallableT>
Thread Resources::make_thread(std::string desc, CpuSet cpu_affinity, CallableT&& callable) const
{
    CHECK(m_thread_resources);
    return m_thread_resources->make_thread(std::move(desc), std::move(cpu_affinity), std::move(callable));
}

}  // namespace mrc::internal::system

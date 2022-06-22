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

#include "internal/system/system.hpp"

#include "internal/system/fiber_task_queue.hpp"
#include "internal/system/partitions.hpp"
#include "public/utils/thread_utils.hpp"

#include <srf/core/bitmap.hpp>
#include <srf/internal/system/isystem.hpp>
#include <srf/options/fiber_pool.hpp>
#include <srf/options/options.hpp>
#include <srf/types.hpp>
#include <srf/utils/string_utils.hpp>

#include <glog/logging.h>
#include <hwloc.h>
#include <boost/fiber/future/future.hpp>

#include <map>
#include <thread>
#include <type_traits>
#include <vector>

namespace srf::internal::system {

std::shared_ptr<System> System::make_system(std::shared_ptr<Options> options)
{
    return std::shared_ptr<System>(new System(std::move(options)));
}

std::shared_ptr<System> System::unwrap(const ISystem& system)
{
    return system.m_impl;
}

System::System(std::shared_ptr<const Options> options) :
  m_options(options),
  m_topology(Topology::Create(options->topology())),
  m_partitions(std::make_shared<Partitions>(*this)),
  m_fiber_manager(std::make_shared<FiberManager>(*this))
{}

System::~System()
{
    DVLOG(10) << "shutting down fiber manager";
    m_fiber_manager.reset();
}

const Options& System::options() const
{
    CHECK(m_options);
    return *m_options;
}

const Topology& System::topology() const
{
    CHECK(m_topology);
    return *m_topology;
}

void System::initialize_thread(const std::string& desc, const CpuSet& cpu_affinity) const
{
    std::string affinity;

    if (cpu_affinity.weight() == 1)
    {
        std::stringstream ss;
        ss << "cpu_id: " << cpu_affinity.first();
        affinity = ss.str();
    }
    else
    {
        std::stringstream ss;
        ss << "cpus: " << cpu_affinity.str();
        affinity      = ss.str();
        auto numa_set = topology().numaset_for_cpuset(cpu_affinity);
        if (numa_set.weight() != 1)
        {
            LOG(FATAL) << "allowing thread to migrate across numa boundaries is currently disabled";
        }
    }

    // todo(ryan) - enable thread/memory binding should be a system option, not specifically a fiber_pool option
    if (options().fiber_pool().enable_thread_binding())
    {
        DVLOG(10) << "tid: " << std::this_thread::get_id() << "; setting cpu affinity to " << affinity;
        auto rc = hwloc_set_cpubind(topology().handle(), &cpu_affinity.bitmap(), HWLOC_CPUBIND_THREAD);
        CHECK_NE(rc, -1);
        set_current_thread_name(SRF_CONCAT_STR("[" << desc << "; " << affinity << "]"));
    }
    else
    {
        DVLOG(10) << "thread_binding is disabled; tid: " << std::this_thread::get_id()
                  << " will use the affinity of caller";
        set_current_thread_name(SRF_CONCAT_STR("[" << desc << "; tid:" << std::this_thread::get_id() << "]"));
    }

    // todo(ryan) - enable thread/memory binding should be a system option, not specifically a fiber_pool option
    if (options().fiber_pool().enable_memory_binding() and m_cap_membind)
    {
        // determine which partition the intended cpu_affinity matches - only one match can exist
        // use the following cpu_set which may span multiple numa nodes to allow membind across all numa domains in the
        // host partition
        // if enabled, this can relax the fatal condition above where we disable the threads ability to migrate across
        // numa boundaries
        // partitions().host_partition_containing(cpu_affinity).cpu_set()

        DVLOG(10) << "tid: " << std::this_thread::get_id()
                  << "; setting memory affinity to the numa_node associated with " << affinity;
        auto rc =
            hwloc_set_membind(topology().handle(), &cpu_affinity.bitmap(), HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD);
        if (rc == -1)
        {
            LOG(WARNING) << "unable to set memory policy - if using docker use: --cap-add=sys_nice to allow membind";
            m_cap_membind = false;
        }
    }

    // use the first cpu_id in the cpu_affinity to query and execute the thread_local initializers
    int cpu_id  = cpu_affinity.first();
    auto result = m_thread_initializers.equal_range(cpu_id);
    for (auto it = result.first; it != result.second; it++)
    {
        it->second();
    }
}

void System::register_thread_local_initializer(const CpuSet& cpu_set, std::function<void()> initializer)
{
    CHECK_GE(cpu_set.weight(), 0);
    CHECK(topology().contains(cpu_set));
    CHECK(initializer);
    cpu_set.for_each_bit([this, initializer](std::uint32_t idx, std::uint32_t bit) {
        m_thread_initializers.insert(std::make_pair(bit, initializer));
    });
    auto futures =
        m_fiber_manager->enqueue_fiber_on_cpuset(cpu_set, [initializer](std::uint32_t cpu_id) { initializer(); });
    for (auto& f : futures)
    {
        f.get();
    }
}

void System::register_thread_local_finalizer(const CpuSet& cpu_set, std::function<void()> initializer)
{
    CHECK_GE(cpu_set.weight(), 0);
    CHECK(topology().contains(cpu_set));
    CHECK(initializer);
    cpu_set.for_each_bit([this, initializer](std::uint32_t idx, std::uint32_t bit) {
        m_thread_finalizers.insert(std::make_pair(bit, initializer));
    });
}

std::shared_ptr<FiberPool> System::make_fiber_pool(CpuSet cpu_set) const
{
    CHECK(m_fiber_manager);
    return m_fiber_manager->make_pool(cpu_set);
}

const Partitions& System::partitions() const
{
    CHECK(m_partitions);
    return *m_partitions;
}

CpuSet System::get_current_thread_affinity() const
{
    CpuSet cpu_set;
    hwloc_get_cpubind(topology().handle(), &cpu_set.bitmap(), HWLOC_CPUBIND_THREAD);
    return cpu_set;
}

std::shared_ptr<FiberTaskQueue> System::get_task_queue(std::uint32_t cpu_id) const
{
    return m_fiber_manager->task_queue(cpu_id);
}

void System::finalize_thread(const CpuSet& cpu_affinity) const
{
    int cpu_id  = cpu_affinity.first();
    auto result = m_thread_finalizers.equal_range(cpu_id);
    for (auto it = result.first; it != result.second; it++)
    {
        it->second();
    }
}

}  // namespace srf::internal::system

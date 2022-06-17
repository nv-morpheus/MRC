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

#include "internal/system/thread.hpp"

#include "internal/system/system.hpp"
#include "internal/system/topology.hpp"
#include "public/utils/thread_utils.hpp"

#include "srf/core/bitmap.hpp"
#include "srf/options/fiber_pool.hpp"
#include "srf/options/options.hpp"
#include "srf/utils/string_utils.hpp"

#include <glog/logging.h>
#include <hwloc.h>

#include <cstdint>
#include <type_traits>

namespace srf::internal::system {

Thread::Thread(std::shared_ptr<const ThreadResources> resources, std::thread&& thread) :
  m_resources(std::move(resources)),
  m_thread(std::move(thread))
{}

Thread::~Thread()
{
    if (m_thread.joinable())
    {
        DVLOG(10) << "[system::Thread]: joining tid=" << m_thread.get_id();
        m_thread.join();
        m_resources.reset();
    }
}

const std::thread& Thread::thread() const
{
    return m_thread;
}

void Thread::join()
{
    m_thread.join();
}

void ThreadResources::register_initializer(const CpuSet& cpu_set, std::function<void()> initializer)
{
    CHECK(initializer);
    CHECK_GE(cpu_set.weight(), 0);
    CHECK(system().topology().contains(cpu_set));
    cpu_set.for_each_bit([this, initializer](std::uint32_t idx, std::uint32_t bit) {
        m_thread_initializers.insert(std::make_pair(bit, initializer));
    });
}

void ThreadResources::register_finalizer(const CpuSet& cpu_set, std::function<void()> finalizer)
{
    CHECK(finalizer);
    CHECK_GE(cpu_set.weight(), 0);
    CHECK(system().topology().contains(cpu_set));
    cpu_set.for_each_bit([this, finalizer](std::uint32_t idx, std::uint32_t bit) {
        m_thread_finalizers.insert(std::make_pair(bit, finalizer));
    });
}

void ThreadResources::initialize_thread(const std::string& desc, const CpuSet& cpu_affinity) const
{
    std::string affinity;

    const auto& options  = system().options();
    const auto& topology = system().topology();

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
        auto numa_set = topology.numaset_for_cpuset(cpu_affinity);
        if (numa_set.weight() != 1)
        {
            LOG(FATAL) << "allowing thread to migrate across numa boundaries is currently disabled";
        }
    }

    // todo(ryan) - enable thread/memory binding should be a system option, not specifically a fiber_pool option
    if (options.fiber_pool().enable_thread_binding())
    {
        DVLOG(10) << "tid: " << std::this_thread::get_id() << "; setting cpu affinity to " << affinity;
        auto rc = hwloc_set_cpubind(topology.handle(), &cpu_affinity.bitmap(), HWLOC_CPUBIND_THREAD);
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
    if (options.fiber_pool().enable_memory_binding() and m_cap_membind)
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
            hwloc_set_membind(topology.handle(), &cpu_affinity.bitmap(), HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_THREAD);
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

void ThreadResources::finalize_thread(const CpuSet& cpu_affinity) const
{
    int cpu_id  = cpu_affinity.first();
    auto result = m_thread_finalizers.equal_range(cpu_id);
    for (auto it = result.first; it != result.second; it++)
    {
        it->second();
    }
}

}  // namespace srf::internal::system

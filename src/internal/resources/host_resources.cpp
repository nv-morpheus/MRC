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

#include "internal/resources/host_resources.hpp"

#include "internal/runnable/engine_factory.hpp"
#include "internal/system/engine_factory_cpu_sets.hpp"
#include "internal/system/system.hpp"

#include "srf/core/bitmap.hpp"
#include "srf/core/task_queue.hpp"
#include "srf/runnable/launch_control.hpp"
#include "srf/runnable/launch_control_config.hpp"
#include "srf/runnable/types.hpp"
#include "srf/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>

#include <map>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

namespace srf::internal::resources {

HostResources::HostResources(std::shared_ptr<system::System> system, const system::HostPartition& partition) :
  m_partition(partition)
{
    DVLOG(10) << "constructing main task queue for host partition " << partition.cpu_set().str();
    auto search = partition.engine_factory_cpu_sets().fiber_cpu_sets.find("main");
    CHECK(search != partition.engine_factory_cpu_sets().fiber_cpu_sets.end()) << "unable to lookup cpuset for main";
    CHECK_EQ(search->second.weight(), 1);
    m_main = system->get_task_queue(search->second.first());

    // construct all other resources on main
    m_main
        ->enqueue([this, system, &partition]() mutable {
            DVLOG(10) << "constructing engine factories on main for host partition " << partition.cpu_set().str();
            ::srf::runnable::LaunchControlConfig config;

            for (const auto& [name, cpu_set] : partition.engine_factory_cpu_sets().fiber_cpu_sets)
            {
                auto reusable = partition.engine_factory_cpu_sets().is_resuable(name);
                DVLOG(10) << "fiber engine factory: " << name << " using " << cpu_set.str() << " is "
                          << (reusable ? "resuable" : "not reusable");
                config.resource_groups[name] =
                    runnable::make_engine_factory(system, runnable::EngineType::Fiber, cpu_set, reusable);
            }

            for (const auto& [name, cpu_set] : partition.engine_factory_cpu_sets().thread_cpu_sets)
            {
                auto reusable = partition.engine_factory_cpu_sets().is_resuable(name);
                DVLOG(10) << "thread engine factory: " << name << " using " << cpu_set.str() << " is "
                          << (reusable ? "resuable" : "not reusable");
                config.resource_groups[name] =
                    runnable::make_engine_factory(system, runnable::EngineType::Thread, cpu_set, reusable);
            }

            // construct launch control
            DVLOG(10) << "constructing launch control on main for host partition " << partition.cpu_set().str();
            m_launch_control = std::make_shared<::srf::runnable::LaunchControl>(std::move(config));

            // construct host memory resource
            DVLOG(10) << "constructing memory_resource on main for host partition " << partition.cpu_set().str()
                      << " - not yet implemeted";
        })
        .get();
}

core::FiberTaskQueue& HostResources::main()
{
    CHECK(m_main);
    return *m_main;
}

::srf::runnable::LaunchControl& HostResources::launch_control()
{
    CHECK(m_launch_control);
    return *m_launch_control;
}

const system::HostPartition& HostResources::partition() const
{
    return m_partition;
}
}  // namespace srf::internal::resources

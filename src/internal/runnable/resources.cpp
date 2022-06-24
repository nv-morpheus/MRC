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

#include "internal/runnable/resources.hpp"

#include <srf/runnable/launch_control_config.hpp>

#include "internal/runnable/engine_factory.hpp"
#include "internal/runnable/engines.hpp"
#include "internal/system/engine_factory_cpu_sets.hpp"
#include "internal/system/host_partition.hpp"
#include "internal/system/partitions.hpp"
#include "internal/system/system.hpp"

#include "srf/core/bitmap.hpp"
#include "srf/runnable/types.hpp"
#include "srf/types.hpp"

#include <glog/logging.h>
#include <boost/fiber/future/future.hpp>

#include <map>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace srf::internal::runnable {

Resources::Resources(const system::Resources& system_resources, std::size_t partition_id) :
  SystemProvider(system_resources),
  m_partition_id(partition_id),
  m_main(system_resources.get_task_queue(this->partition().host().engine_factory_cpu_sets().main_cpu_id()))
{
    CHECK_LT(partition_id, system().partitions().flattened().size());
    const auto& partition = this->partition();

    DVLOG(10) << "main fiber task queue for host partition " << m_partition_id << " assigned to cpu_id "
              << partition.host().engine_factory_cpu_sets().main_cpu_id();

    // construct all other resources on main
    main()
        .enqueue([this, &system_resources, &partition]() mutable {
            DVLOG(10) << "constructing engine factories on main for host partition "
                      << partition.host().cpu_set().str();
            ::srf::runnable::LaunchControlConfig config;

            for (const auto& [name, cpu_set] : partition.host().engine_factory_cpu_sets().fiber_cpu_sets)
            {
                auto reusable = partition.host().engine_factory_cpu_sets().is_resuable(name);
                DVLOG(10) << "fiber engine factory: " << name << " using " << cpu_set.str() << " is "
                          << (reusable ? "resuable" : "not reusable");
                config.resource_groups[name] =
                    runnable::make_engine_factory(system_resources, runnable::EngineType::Fiber, cpu_set, reusable);
            }

            for (const auto& [name, cpu_set] : partition.host().engine_factory_cpu_sets().thread_cpu_sets)
            {
                auto reusable = partition.host().engine_factory_cpu_sets().is_resuable(name);
                DVLOG(10) << "thread engine factory: " << name << " using " << cpu_set.str() << " is "
                          << (reusable ? "resuable" : "not reusable");
                config.resource_groups[name] =
                    runnable::make_engine_factory(system_resources, runnable::EngineType::Thread, cpu_set, reusable);
            }

            // construct launch control
            DVLOG(10) << "constructing launch control on main for host partition " << partition.host().cpu_set().str();
            m_launch_control = std::make_unique<::srf::runnable::LaunchControl>(std::move(config));

            // construct host memory resource
            DVLOG(10) << "constructing memory_resource on main for host partition " << partition.host().cpu_set().str()
                      << " - not yet implemeted";
        })
        .get();
}

core::FiberTaskQueue& Resources::main()
{
    return m_main;
}

srf::runnable::LaunchControl& Resources::launch_control()
{
    CHECK(m_launch_control);
    return *m_launch_control;
}

const system::Partition& Resources::partition() const
{
    return system().partitions().flattened().at(m_partition_id);
}

std::size_t Resources::partition_id() const
{
    return m_partition_id;
}
}  // namespace srf::internal::runnable

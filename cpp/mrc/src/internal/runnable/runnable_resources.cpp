/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/runnable/runnable_resources.hpp"

#include "internal/runnable/engine_factory.hpp"
#include "internal/system/engine_factory_cpu_sets.hpp"
#include "internal/system/fiber_task_queue.hpp"
#include "internal/system/host_partition.hpp"
#include "internal/system/partitions.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/runnable/launch_control_config.hpp"
#include "mrc/runnable/types.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>

#include <map>
#include <ostream>
#include <string>
#include <utility>

namespace mrc::runnable {

RunnableResources::RunnableResources(const system::ThreadingResources& system_resources,
                                     const system::HostPartition& host_partition) :
  HostPartitionProvider(system_resources, 0),
  m_main(system_resources.get_task_queue(host_partition.engine_factory_cpu_sets().main_cpu_id())),
  m_network(system_resources.get_task_queue(
      host_partition.engine_factory_cpu_sets().fiber_cpu_sets.at("mrc_network").first()))
{
    DVLOG(10) << "main fiber task queue for host partition " << host_partition_id() << " assigned to cpu_id "
              << host_partition.engine_factory_cpu_sets().main_cpu_id();

    // construct all other resources on main
    main()
        .enqueue([this, &system_resources, &host_partition]() mutable {
            DVLOG(10) << "constructing engine factories on main for host partition " << host_partition.cpu_set().str();
            mrc::runnable::LaunchControlConfig config;

            for (const auto& [name, cpu_set] : host_partition.engine_factory_cpu_sets().fiber_cpu_sets)
            {
                auto reusable = host_partition.engine_factory_cpu_sets().is_resuable(name);
                DVLOG(10) << "fiber engine factory: " << name << " using " << cpu_set.str() << " is "
                          << (reusable ? "resuable" : "not reusable");
                config.resource_groups[name] =
                    runnable::make_engine_factory(system_resources, runnable::EngineType::Fiber, cpu_set, reusable);
            }

            for (const auto& [name, cpu_set] : host_partition.engine_factory_cpu_sets().thread_cpu_sets)
            {
                auto reusable = host_partition.engine_factory_cpu_sets().is_resuable(name);
                DVLOG(10) << "thread engine factory: " << name << " using " << cpu_set.str() << " is "
                          << (reusable ? "resuable" : "not reusable");
                config.resource_groups[name] =
                    runnable::make_engine_factory(system_resources, runnable::EngineType::Thread, cpu_set, reusable);
            }

            // construct launch control
            DVLOG(10) << "constructing launch control on main for host partition " << host_partition.cpu_set().str();
            m_launch_control = std::make_unique<::mrc::runnable::LaunchControl>(std::move(config));
        })
        .get();
}

const system::HostPartition& find_host_partition(const system::SystemProvider& system_provider,
                                                 std::size_t host_partition_id)
{
    CHECK_LT(host_partition_id, system_provider.system().partitions().host_partitions().size());

    return system_provider.system().partitions().host_partitions().at(host_partition_id);
}

RunnableResources::RunnableResources(const system::ThreadingResources& system_resources,
                                     std::size_t _host_partition_id) :
  RunnableResources(system_resources, find_host_partition(system_resources, _host_partition_id))
{
    // DVLOG(10) << "main fiber task queue for host partition " << host_partition_id() << " assigned to cpu_id "
    //           << host_partition.engine_factory_cpu_sets().main_cpu_id();

    // // construct all other resources on main
    // main()
    //     .enqueue([this, &system_resources, &host_partition]() mutable {
    //         DVLOG(10) << "constructing engine factories on main for host partition " <<
    //         host_partition.cpu_set().str(); mrc::runnable::LaunchControlConfig config;

    //         for (const auto& [name, cpu_set] : host_partition.engine_factory_cpu_sets().fiber_cpu_sets)
    //         {
    //             auto reusable = host_partition.engine_factory_cpu_sets().is_resuable(name);
    //             DVLOG(10) << "fiber engine factory: " << name << " using " << cpu_set.str() << " is "
    //                       << (reusable ? "resuable" : "not reusable");
    //             config.resource_groups[name] =
    //                 runnable::make_engine_factory(system_resources, runnable::EngineType::Fiber, cpu_set, reusable);
    //         }

    //         for (const auto& [name, cpu_set] : host_partition.engine_factory_cpu_sets().thread_cpu_sets)
    //         {
    //             auto reusable = host_partition.engine_factory_cpu_sets().is_resuable(name);
    //             DVLOG(10) << "thread engine factory: " << name << " using " << cpu_set.str() << " is "
    //                       << (reusable ? "resuable" : "not reusable");
    //             config.resource_groups[name] =
    //                 runnable::make_engine_factory(system_resources, runnable::EngineType::Thread, cpu_set, reusable);
    //         }

    //         // construct launch control
    //         DVLOG(10) << "constructing launch control on main for host partition " << host_partition.cpu_set().str();
    //         m_launch_control = std::make_unique<::mrc::runnable::LaunchControl>(std::move(config));
    //     })
    //     .get();
}

RunnableResources::RunnableResources(RunnableResources&& other) = default;

RunnableResources::~RunnableResources() = default;

core::FiberTaskQueue& RunnableResources::main()
{
    return m_main;
}

const mrc::core::FiberTaskQueue& RunnableResources::main() const
{
    return m_main;
}

mrc::core::FiberTaskQueue& RunnableResources::network()
{
    return m_network;
}

const mrc::core::FiberTaskQueue& RunnableResources::network() const
{
    return m_network;
}

mrc::runnable::LaunchControl& RunnableResources::launch_control()
{
    CHECK(m_launch_control);
    return *m_launch_control;
}

}  // namespace mrc::runnable

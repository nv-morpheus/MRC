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

#include "internal/runnable/engine_factory.hpp"
#include "internal/runnable/engines.hpp"
#include "internal/system/engine_factory_cpu_sets.hpp"
#include "internal/system/host_partition.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/runnable/launch_control_config.hpp"
#include "mrc/runnable/types.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>

#include <map>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

namespace mrc::internal::runnable {

Resources::Resources(const system::Resources& system_resources, std::size_t _host_partition_id) :
  HostPartitionProvider(system_resources, _host_partition_id),
  m_main(system_resources.get_task_queue(host_partition().engine_factory_cpu_sets().main_cpu_id()))
{
    const auto& host_partition = this->host_partition();

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

core::FiberTaskQueue& Resources::main()
{
    return m_main;
}

mrc::runnable::LaunchControl& Resources::launch_control()
{
    CHECK(m_launch_control);
    return *m_launch_control;
}

const mrc::core::FiberTaskQueue& Resources::main() const
{
    return m_main;
}
}  // namespace mrc::internal::runnable

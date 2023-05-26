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

#include "mrc/runnable/runnable_resources.hpp"

#include "internal/runnable/engine_factory.hpp"
#include "internal/system/engine_factory_cpu_sets.hpp"
#include "internal/system/fiber_task_queue.hpp"
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
#include <utility>

namespace mrc::runnable {

const mrc::core::FiberTaskQueue& IRunnableResources::main() const
{
    return const_cast<IRunnableResources*>(this)->main();
}

const IRunnableResources& IRunnableResourcesProvider::runnable() const
{
    // Return the other overload
    return const_cast<IRunnableResourcesProvider*>(this)->runnable();
}

RunnableResourcesProvider RunnableResourcesProvider::create(IRunnableResources& runnable)
{
    return {runnable};
}

RunnableResourcesProvider::RunnableResourcesProvider(const RunnableResourcesProvider& other) :
  m_runnable(other.m_runnable)
{}

RunnableResourcesProvider::RunnableResourcesProvider(IRunnableResourcesProvider& other) : m_runnable(other.runnable())
{}

RunnableResourcesProvider::RunnableResourcesProvider(IRunnableResources& runnable) : m_runnable(runnable) {}

IRunnableResources& RunnableResourcesProvider::runnable()
{
    return m_runnable;
}

}  // namespace mrc::runnable

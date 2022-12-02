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

#include "internal/runnable/thread_engines.hpp"

#include "internal/runnable/thread_engine.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/types.hpp"

#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

namespace mrc::internal::runnable {

void ThreadEngines::initialize_launchers()
{
    CHECK_EQ(launch_options().pe_count, m_cpu_set.weight())
        << "mismatch in the number of cores in the cpu set with respect to the requested pe_count";
    CHECK_GE(launch_options().engines_per_pe, 1);

    m_cpu_set.for_each_bit([this](std::uint32_t idx, std::uint32_t cpu_id) {
        CpuSet cpu;
        cpu.only(cpu_id);
        for (int i = 0; i < launch_options().engines_per_pe; ++i)
        {
            add_launcher(std::make_shared<ThreadEngine>(cpu, m_system));
        }
    });
}

ThreadEngines::ThreadEngines(CpuSet cpu_set, const system::Resources& system) :
  ThreadEngines(LaunchOptions("custom_options", cpu_set.weight()), cpu_set, std::move(system))
{}

ThreadEngines::ThreadEngines(LaunchOptions launch_options, CpuSet cpu_set, const system::Resources& system) :
  Engines(std::move(launch_options)),
  m_cpu_set(std::move(cpu_set)),
  m_system(std::move(system))
{
    initialize_launchers();
}

runnable::EngineType ThreadEngines::engine_type() const
{
    return EngineType::Thread;
}
}  // namespace mrc::internal::runnable

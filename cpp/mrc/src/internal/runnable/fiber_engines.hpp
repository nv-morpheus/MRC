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

#pragma once

#include "internal/runnable/engine.hpp"
#include "internal/runnable/engines.hpp"
#include "internal/system/fiber_pool.hpp"

#include "mrc/constants.hpp"
#include "mrc/core/fiber_meta_data.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/types.hpp"

#include <functional>
#include <vector>

namespace mrc::internal::runnable {

class FiberEngines final : public Engines
{
  public:
    FiberEngines(system::FiberPool& pool, int priority = MRC_DEFAULT_FIBER_PRIORITY);

    FiberEngines(::mrc::runnable::LaunchOptions launch_options,
                 system::FiberPool& pool,
                 int priority = MRC_DEFAULT_FIBER_PRIORITY);

    FiberEngines(::mrc::runnable::LaunchOptions launch_options, system::FiberPool& pool, const FiberMetaData& meta);

    FiberEngines(::mrc::runnable::LaunchOptions launch_options,
                 std::vector<std::reference_wrapper<core::FiberTaskQueue>>&& task_queues,
                 int priority = MRC_DEFAULT_FIBER_PRIORITY);

    ~FiberEngines() final = default;

    EngineType engine_type() const final;

  private:
    void initialize_launchers();

    std::vector<std::reference_wrapper<mrc::core::FiberTaskQueue>> m_task_queues;
    FiberMetaData m_meta;
};

}  // namespace mrc::internal::runnable

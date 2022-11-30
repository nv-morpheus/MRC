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

#include "internal/runnable/engines.hpp"
#include "internal/system/resources.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/types.hpp"

namespace mrc::internal::runnable {

class ThreadEngines final : public Engines
{
  public:
    ThreadEngines(CpuSet cpu_set, const system::Resources& system);
    ThreadEngines(LaunchOptions launch_options, CpuSet cpu_set, const system::Resources& system);
    ~ThreadEngines() final = default;

    EngineType engine_type() const final;

  private:
    void initialize_launchers();

    CpuSet m_cpu_set;
    const system::Resources& m_system;
};

}  // namespace mrc::internal::runnable

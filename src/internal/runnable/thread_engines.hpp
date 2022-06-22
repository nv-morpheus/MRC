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

#include "internal/system/system.hpp"

#include <srf/core/bitmap.hpp>
#include <srf/runnable/launch_options.hpp>
#include <srf/runnable/types.hpp>
#include <srf/types.hpp>

#include <memory>

namespace srf::internal::runnable {

class ThreadEngines final : public Engines
{
  public:
    ThreadEngines(CpuSet cpu_set, std::shared_ptr<system::System> system);
    ThreadEngines(LaunchOptions launch_options, CpuSet cpu_set, std::shared_ptr<system::System> system);
    ~ThreadEngines() final = default;

    EngineType engine_type() const final;

  private:
    void initialize_launchers();

    CpuSet m_cpu_set;
    Handle<system::System> m_system;
};

}  // namespace srf::internal::runnable

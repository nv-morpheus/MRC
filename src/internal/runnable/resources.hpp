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

#include "internal/system/fiber_task_queue.hpp"
#include "internal/system/host_partition_provider.hpp"
#include "internal/system/resources.hpp"

#include "srf/core/task_queue.hpp"
#include "srf/pipeline/resources.hpp"
#include "srf/runnable/launch_control.hpp"

#include <cstddef>
#include <memory>

namespace srf::internal::runnable {

class Resources final : public system::HostPartitionProvider, public srf::pipeline::Resources
{
  public:
    Resources(const system::Resources& system_resources, std::size_t _host_partition_id);

    srf::core::FiberTaskQueue& main() final;
    const srf::core::FiberTaskQueue& main() const;
    srf::runnable::LaunchControl& launch_control() final;

  private:
    system::FiberTaskQueue& m_main;
    std::unique_ptr<srf::runnable::LaunchControl> m_launch_control;
};

}  // namespace srf::internal::runnable

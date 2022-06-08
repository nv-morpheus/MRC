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

#include "internal/system/forward.hpp"
#include "internal/system/host_partition.hpp"
#include "srf/core/task_queue.hpp"
#include "srf/pipeline/resources.hpp"
#include "srf/runnable/launch_control.hpp"

#include <memory>

namespace srf::internal::resources {

class HostResources : public ::srf::pipeline::Resources
{
  public:
    HostResources(std::shared_ptr<system::System> system, const system::HostPartition& partition);

    const system::HostPartition& partition() const;
    ::srf::core::FiberTaskQueue& main() final;
    ::srf::runnable::LaunchControl& launch_control() final;

  private:
    const system::HostPartition& m_partition;
    std::shared_ptr<::srf::core::FiberTaskQueue> m_main;
    std::shared_ptr<::srf::runnable::LaunchControl> m_launch_control;
};

}  // namespace srf::internal::resources

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/system/host_partition_provider.hpp"
#include "internal/system/threading_resources.hpp"

#include "mrc/core/task_queue.hpp"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/runnable_resources.hpp"

#include <cstddef>
#include <memory>

namespace mrc::system {
class FiberTaskQueue;
}  // namespace mrc::system

namespace mrc::runnable {

class RunnableResources final : public system::HostPartitionProvider, public IRunnableResources
{
  public:
    RunnableResources(const system::ThreadingResources& system_resources, std::size_t _host_partition_id);
    RunnableResources(RunnableResources&& other);
    ~RunnableResources() override;

    mrc::core::FiberTaskQueue& main() final;
    const mrc::core::FiberTaskQueue& main() const;
    mrc::runnable::LaunchControl& launch_control() final;

  private:
    system::FiberTaskQueue& m_main;
    std::unique_ptr<mrc::runnable::LaunchControl> m_launch_control;
};

}  // namespace mrc::runnable

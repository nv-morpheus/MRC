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

#pragma once

#include "internal/system/host_partition.hpp"
#include "internal/system/host_partition_provider.hpp"
#include "internal/system/threading_resources.hpp"

#include "mrc/core/task_queue.hpp"
#include "mrc/pipeline/resources.hpp"
#include "mrc/runnable/launch_control.hpp"

#include <cstddef>
#include <memory>

namespace mrc::internal::system {
class FiberTaskQueue;
}  // namespace mrc::internal::system

namespace mrc::internal::runnable {

class RunnableResources final : public system::HostPartitionProvider, public mrc::pipeline::Resources
{
  public:
    RunnableResources(const system::ThreadingResources& system_resources, std::size_t _host_partition_id);
    RunnableResources(const system::ThreadingResources& system_resources, const system::HostPartition& host_partition);
    RunnableResources(RunnableResources&& other);
    ~RunnableResources() override;

    mrc::core::FiberTaskQueue& main() final;
    const mrc::core::FiberTaskQueue& main() const;
    mrc::runnable::LaunchControl& launch_control() final;

  private:
    system::FiberTaskQueue& m_main;
    std::unique_ptr<mrc::runnable::LaunchControl> m_launch_control;
};

class IRunnableResourcesProvider
{
  protected:
    virtual RunnableResources& runnable() = 0;

    const RunnableResources& runnable() const;

  private:
    friend class RunnableResourcesProvider;
};

// Concrete implementation of IRunnableResourcesProvider. Use this if RunnableResources is available during
// construction. Inherits virtually to ensure only one IRunnableResourcesProvider
class RunnableResourcesProvider : public virtual IRunnableResourcesProvider
{
  protected:
    RunnableResourcesProvider(const RunnableResourcesProvider& other);
    RunnableResourcesProvider(IRunnableResourcesProvider& other);
    RunnableResourcesProvider(RunnableResources& runnable);

    RunnableResources& runnable() override;

  private:
    RunnableResources& m_runnable;
};

}  // namespace mrc::internal::runnable

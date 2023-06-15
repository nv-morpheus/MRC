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

namespace mrc::core {
class FiberTaskQueue;
}

namespace mrc::runnable {
class LaunchControl;

struct IRunnableResources
{
    virtual ~IRunnableResources() = default;

    virtual core::FiberTaskQueue& main() = 0;
    const core::FiberTaskQueue& main() const;

    virtual runnable::LaunchControl& launch_control() = 0;

    // virtual std::shared_ptr<metrics::Registry> metrics_registry() = 0;
};

class IRunnableResourcesProvider
{
  protected:
    virtual IRunnableResources& runnable() = 0;

    const IRunnableResources& runnable() const;

  private:
    friend class RunnableResourcesProvider;
};

// Concrete implementation of IRunnableResourcesProvider. Use this if RunnableResources is available during
// construction. Inherits virtually to ensure only one IRunnableResourcesProvider
class RunnableResourcesProvider : public virtual IRunnableResourcesProvider
{
  public:
    static RunnableResourcesProvider create(IRunnableResources& runnable);

  protected:
    RunnableResourcesProvider(const RunnableResourcesProvider& other);
    RunnableResourcesProvider(IRunnableResourcesProvider& other);
    RunnableResourcesProvider(IRunnableResources& runnable);

    IRunnableResources& runnable() override;

  private:
    IRunnableResources& m_runnable;
};

}  // namespace mrc::runnable

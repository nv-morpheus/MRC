/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/runnable/runnable_resources.hpp"

#include <cstddef>

namespace mrc::control_plane {
class Client;
}

namespace mrc::metrics {
class Registry;
}

namespace mrc::runtime {

class PipelinesManager;
class DataPlaneSystemManager;

struct IInternalRuntime
{
    virtual ~IInternalRuntime() = default;

    // GPUs available to this runtime object
    virtual size_t gpu_count() const = 0;

    virtual runnable::IRunnableResources& runnable() = 0;

    virtual control_plane::Client& control_plane() const = 0;

    virtual DataPlaneSystemManager& data_plane() const = 0;

    virtual PipelinesManager& pipelines_manager() const = 0;

    virtual metrics::Registry& metrics_registry() const = 0;
};

class IInternalRuntimeProvider : public virtual runnable::IRunnableResourcesProvider
{
  protected:
    virtual IInternalRuntime& runtime() = 0;

    const IInternalRuntime& runtime() const;

    runnable::IRunnableResources& runnable() override;

  private:
    friend class InternalRuntimeProvider;
};

// Concrete implementation of IInternalRuntimeProvider. Use this if IInternalRuntime is available during
// construction. Inherits virtually to ensure only one IInternalRuntimeProvider
class InternalRuntimeProvider : public virtual IInternalRuntimeProvider
{
  public:
    using interface_t = IInternalRuntimeProvider;

    static InternalRuntimeProvider create(IInternalRuntime& runtime);

  protected:
    InternalRuntimeProvider(const InternalRuntimeProvider& other);
    InternalRuntimeProvider(IInternalRuntimeProvider& other);
    InternalRuntimeProvider(IInternalRuntime& runtime);

    IInternalRuntime& runtime() override;

  private:
    IInternalRuntime& m_runtime;
};

struct IInternalPartitionRuntime : public IInternalRuntime
{
    ~IInternalPartitionRuntime() override = default;

    virtual size_t partition_id() const = 0;

    // virtual DataPlaneSystemManager& data_plane() const = 0;
};

class IInternalPartitionRuntimeProvider : public virtual IInternalRuntimeProvider
{
  protected:
    IInternalPartitionRuntime& runtime() override = 0;

    const IInternalPartitionRuntime& runtime() const;

  private:
    friend class InternalPartitionRuntimeProvider;
};

// Concrete implementation of IInternalPartitionRuntimeProvider. Use this if IInternalPartitionRuntime is available
// during construction. Inherits virtually to ensure only one IInternalPartitionRuntimeProvider
class InternalPartitionRuntimeProvider : public virtual IInternalPartitionRuntimeProvider
{
  public:
    using interface_t = IInternalPartitionRuntimeProvider;

    static InternalPartitionRuntimeProvider create(IInternalPartitionRuntime& runtime);

  protected:
    InternalPartitionRuntimeProvider(const InternalPartitionRuntimeProvider& other);
    InternalPartitionRuntimeProvider(IInternalPartitionRuntimeProvider& other);
    InternalPartitionRuntimeProvider(IInternalPartitionRuntime& runtime);

    IInternalPartitionRuntime& runtime() override;

  private:
    IInternalPartitionRuntime& m_runtime;
};

}  // namespace mrc::runtime

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

#include "internal/runtime/runtime_provider.hpp"

namespace mrc::runtime {

const IInternalRuntime& IInternalRuntimeProvider::runtime() const
{
    // Return the other overload
    return const_cast<IInternalRuntimeProvider*>(this)->runtime();
}

runnable::IRunnableResources& IInternalRuntimeProvider::runnable()
{
    return this->runtime().runnable();
}

InternalRuntimeProvider InternalRuntimeProvider::create(IInternalRuntime& runtime)
{
    return {runtime};
}

InternalRuntimeProvider::InternalRuntimeProvider(const InternalRuntimeProvider& other) : m_runtime(other.m_runtime) {}

InternalRuntimeProvider::InternalRuntimeProvider(IInternalRuntimeProvider& other) : m_runtime(other.runtime()) {}

InternalRuntimeProvider::InternalRuntimeProvider(IInternalRuntime& runtime) : m_runtime(runtime) {}

IInternalRuntime& InternalRuntimeProvider::runtime()
{
    return m_runtime;
}

const IInternalPartitionRuntime& IInternalPartitionRuntimeProvider::runtime() const
{
    // Return the other overload
    return const_cast<IInternalPartitionRuntimeProvider*>(this)->runtime();
}

InternalPartitionRuntimeProvider InternalPartitionRuntimeProvider::create(IInternalPartitionRuntime& runtime)
{
    return {runtime};
}

InternalPartitionRuntimeProvider::InternalPartitionRuntimeProvider(const InternalPartitionRuntimeProvider& other) :
  m_runtime(other.m_runtime)
{}

InternalPartitionRuntimeProvider::InternalPartitionRuntimeProvider(IInternalPartitionRuntimeProvider& other) :
  m_runtime(other.runtime())
{}

InternalPartitionRuntimeProvider::InternalPartitionRuntimeProvider(IInternalPartitionRuntime& runtime) :
  m_runtime(runtime)
{}

IInternalPartitionRuntime& InternalPartitionRuntimeProvider::runtime()
{
    return m_runtime;
}

}  // namespace mrc::runtime

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

#include "internal/system/system.hpp"

#include "internal/system/partitions.hpp"
#include "internal/system/topology.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/options/options.hpp"

#include <glog/logging.h>
#include <hwloc.h>

#include <memory>
#include <utility>

namespace mrc::system {

SystemDefinition::SystemDefinition(const Options& options) :
  m_options(std::make_unique<Options>(options)),  // Run the copy constructor to make a copy
  m_topology(Topology::Create(m_options->topology())),
  m_partitions(std::make_shared<Partitions>(*this))
{}

SystemDefinition::SystemDefinition(std::shared_ptr<Options> options) : SystemDefinition(*options) {}

SystemDefinition::~SystemDefinition() = default;

std::unique_ptr<SystemDefinition> SystemDefinition::unwrap(std::unique_ptr<ISystem> object)
{
    // Convert to the full implementation
    auto* full_object_ptr = dynamic_cast<SystemDefinition*>(object.get());

    CHECK(full_object_ptr) << "Invalid cast for SystemDefinition. Please report to the developers";

    // At this point, the object is a valid cast. Release the pointer so it doesnt get deallocated
    object.release();

    return std::unique_ptr<SystemDefinition>(full_object_ptr);
}

const Options& SystemDefinition::options() const
{
    CHECK(m_options);
    return *m_options;
}

const Topology& SystemDefinition::topology() const
{
    CHECK(m_topology);
    return *m_topology;
}

const Partitions& SystemDefinition::partitions() const
{
    CHECK(m_partitions);
    return *m_partitions;
}

void SystemDefinition::add_thread_initializer(std::function<void()> initializer_fn)
{
    m_thread_initializers.emplace_back(std::move(initializer_fn));
}

void SystemDefinition::add_thread_finalizer(std::function<void()> finalizer_fn)
{
    m_thread_finalizers.emplace_back(std::move(finalizer_fn));
}

const std::vector<std::function<void()>>& SystemDefinition::thread_initializers() const
{
    return m_thread_initializers;
}

const std::vector<std::function<void()>>& SystemDefinition::thread_finalizers() const
{
    return m_thread_finalizers;
}

CpuSet SystemDefinition::get_current_thread_affinity() const
{
    CpuSet cpu_set;
    hwloc_get_cpubind(topology().handle(), &cpu_set.bitmap(), HWLOC_CPUBIND_THREAD);
    return cpu_set;
}

}  // namespace mrc::system

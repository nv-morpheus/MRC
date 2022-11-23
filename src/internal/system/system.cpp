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

#include "internal/system/system.hpp"

#include "internal/system/partitions.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/engine/system/isystem.hpp"
#include "mrc/options/options.hpp"

#include <glog/logging.h>
#include <hwloc.h>

namespace mrc::internal::system {

std::shared_ptr<System> System::create(std::shared_ptr<Options> options)
{
    return std::shared_ptr<System>(new System(std::move(options)));
}

std::shared_ptr<System> System::unwrap(const ISystem& system)
{
    return system.m_impl;
}

System::System(std::shared_ptr<Options> options) :
  m_options(options),
  m_topology(Topology::Create(options->topology())),
  m_partitions(std::make_shared<Partitions>(*this))
{}

const Options& System::options() const
{
    CHECK(m_options);
    return *m_options;
}

const Topology& System::topology() const
{
    CHECK(m_topology);
    return *m_topology;
}

const Partitions& System::partitions() const
{
    CHECK(m_partitions);
    return *m_partitions;
}

CpuSet System::get_current_thread_affinity() const
{
    CpuSet cpu_set;
    hwloc_get_cpubind(topology().handle(), &cpu_set.bitmap(), HWLOC_CPUBIND_THREAD);
    return cpu_set;
}

}  // namespace mrc::internal::system

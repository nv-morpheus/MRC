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

#include "mrc/options/topology.hpp"

#include "mrc/core/bitmap.hpp"  // for CpuSet

#include <utility>  // for move

namespace mrc {

// Topology Options

TopologyOptions& TopologyOptions::use_process_cpuset(bool default_true)
{
    m_use_process_cpuset = default_true;
    return *this;
}
TopologyOptions& TopologyOptions::restrict_numa_domains(bool yn)
{
    m_restrict_numa_domains = yn;
    return *this;
}
TopologyOptions& TopologyOptions::restrict_gpus(bool default_false)
{
    m_restrict_gpus = default_false;
    return *this;
}
TopologyOptions& TopologyOptions::user_cpuset(CpuSet&& cpu_set)
{
    m_user_cpuset = std::move(cpu_set);
    return *this;
}
TopologyOptions& TopologyOptions::user_cpuset(std::string cpustr)
{
    CpuSet cpu_set(cpustr);
    m_user_cpuset = std::move(cpu_set);
    return *this;
}
bool TopologyOptions::use_process_cpuset() const
{
    return m_use_process_cpuset;
}
bool TopologyOptions::restrict_numa_domains() const
{
    return m_restrict_numa_domains;
}
bool TopologyOptions::restrict_gpus() const
{
    return m_restrict_gpus;
}
const CpuSet& TopologyOptions::user_cpuset() const
{
    return m_user_cpuset;
}

bool TopologyOptions::ignore_dgx_display() const
{
    return m_ignore_dgx_display;
}
TopologyOptions& TopologyOptions::ignore_dgx_display(bool default_true)
{
    m_ignore_dgx_display = default_true;
    return *this;
}
}  // namespace mrc

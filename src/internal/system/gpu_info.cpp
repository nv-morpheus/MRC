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

#include "internal/system/gpu_info.hpp"

#include "mrc/core/bitmap.hpp"

namespace mrc::internal::system {

const CpuSet& GpuInfo::cpu_set() const
{
    return m_cpu_set;
}
const std::string& GpuInfo::name() const
{
    return m_name;
}
const std::string& GpuInfo::uuid() const
{
    return m_uuid;
}
const std::string& GpuInfo::cpustr() const
{
    return m_cpustr;
}
const std::string& GpuInfo::pcie_bus_id() const
{
    return m_pcie_bus_id;
}
const std::uint64_t& GpuInfo::memory_capacity() const
{
    return m_memory_capacity;
}
int GpuInfo::cuda_device_id() const
{
    return m_cuda_device_id;
}
protos::GpuInfo GpuInfo::serialize() const
{
    protos::GpuInfo info;
    info.set_cpu_set(m_cpustr);
    info.set_name(m_name);
    info.set_uuid(m_uuid);
    info.set_pcie_bus_id(m_pcie_bus_id);
    info.set_memory_capacity(m_memory_capacity);
    info.set_cuda_device_id(m_cuda_device_id);
    return info;
}

GpuInfo GpuInfo::deserialize(const protos::GpuInfo& msg)
{
    GpuInfo info;

    info.m_cpu_set         = CpuSet(msg.cpu_set());
    info.m_cpustr          = msg.cpu_set();
    info.m_name            = msg.name();
    info.m_uuid            = msg.uuid();
    info.m_pcie_bus_id     = msg.pcie_bus_id();
    info.m_memory_capacity = msg.memory_capacity();
    info.m_cuda_device_id  = msg.cuda_device_id();

    return info;
}

}  // namespace mrc::internal::system

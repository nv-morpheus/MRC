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

#include "internal/system/device_partition.hpp"

#include <glog/logging.h>

#include <utility>

namespace mrc::internal::system {

DevicePartition::DevicePartition(const GpuInfo& gpu_info, std::shared_ptr<const HostPartition> host_partition) :
  GpuInfo(gpu_info),
  m_host_partition(std::move(host_partition))
{
    CHECK(m_host_partition);
}

int DevicePartition::cuda_device_id() const
{
    return GpuInfo::cuda_device_id();
}
std::size_t DevicePartition::device_memory_capacity() const
{
    return GpuInfo::memory_capacity();
}
const std::string& DevicePartition::name() const
{
    return GpuInfo::name();
}
const std::string& DevicePartition::uuid() const
{
    return GpuInfo::uuid();
}
const std::string& DevicePartition::pcie_bus_id() const
{
    return GpuInfo::pcie_bus_id();
}
const HostPartition& DevicePartition::host() const
{
    CHECK(m_host_partition);
    return *m_host_partition;
}
}  // namespace mrc::internal::system

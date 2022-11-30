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

#pragma once

#include "internal/system/gpu_info.hpp"
#include "internal/system/host_partition.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace mrc::internal::system {

class DevicePartition final : private GpuInfo
{
  public:
    DevicePartition(const GpuInfo& gpu_info, std::shared_ptr<const HostPartition> host_partition);
    virtual ~DevicePartition() = default;

    int cuda_device_id() const;
    std::size_t device_memory_capacity() const;

    const std::string& name() const;
    const std::string& uuid() const;
    const std::string& pcie_bus_id() const;

    const HostPartition& host() const;

    // memory resource
    // virtual memory::resource memory_resource() = 0;

  private:
    std::shared_ptr<const HostPartition> m_host_partition;
};

}  // namespace mrc::internal::system

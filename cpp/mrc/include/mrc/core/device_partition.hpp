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

#include "mrc/core/bitmap.hpp"
#include "mrc/core/host_partition.hpp"

namespace mrc::core {

struct DevicePartition : public virtual HostPartition
{
    ~DevicePartition() override = default;

    [[nodiscard]] virtual int cuda_device_id() const                 = 0;
    [[nodiscard]] virtual std::size_t device_memory_capacity() const = 0;

    [[nodiscard]] virtual const std::string& name() const        = 0;
    [[nodiscard]] virtual const std::string& uuid() const        = 0;
    [[nodiscard]] virtual const std::string& pcie_bus_id() const = 0;

    // memory resource
    // virtual memory::resource memory_resource() = 0;

  private:
    using HostPartition::device_partition_ids;
};

}  // namespace mrc::core

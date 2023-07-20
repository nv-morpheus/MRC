/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstddef>
#include <set>
#include <string>

// DO NOT INCLUDE <nvml.h>!!!!

// NOLINTBEGIN(readability-identifier-naming)
using hwloc_cpuset_t   = struct hwloc_bitmap_s*;
using hwloc_topology_t = struct hwloc_topology*;
using nvmlDevice_t     = struct nvmlDevice_st*;
// NOLINTEND(readability-identifier-naming)

namespace mrc::system {

struct DeviceInfo
{
    // NOLINTBEGIN(readability-identifier-naming)
    static auto AccessibleDeviceCount() -> std::size_t;
    static auto AccessibleDeviceIndexes() -> std::set<unsigned int>;
    static auto Alignment() -> std::size_t;
    static auto DeviceTotalMemory(unsigned int device_id) -> unsigned long long;
    static auto EnergyConsumption(unsigned int device_id) -> double;
    static auto GetDeviceCpuset(hwloc_topology_t topology, unsigned int device_id, hwloc_cpuset_t set) -> int;
    static auto Name(unsigned int device_id) -> std::string;
    static auto PCIeBusID(unsigned int device_id) -> std::string;
    static auto PowerLimit(unsigned int device_id) -> double;
    static auto PowerUsage(unsigned int device_id) -> double;
    static auto UUID(unsigned int device_id) -> std::string;
    // NOLINTEND(readability-identifier-naming)
};

}  // namespace mrc::system

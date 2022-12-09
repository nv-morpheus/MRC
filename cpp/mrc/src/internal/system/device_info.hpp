/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvml.h>

#include <cstddef>
#include <set>
#include <string>

namespace mrc::internal::system {

struct DeviceInfo
{
    static nvmlDevice_t GetHandleById(unsigned int device_id);  // NOLINT
    // static auto Affinity(int device_id) -> cpu_set;
    static auto Alignment() -> std::size_t;                           // NOLINT
    static auto EnergyConsumption(int device_id) -> double;           // NOLINT
    static auto MemoryInfo(int device_id) -> nvmlMemory_t;            // NOLINT
    static auto PowerUsage(int device_id) -> double;                  // NOLINT
    static auto PowerLimit(int device_id) -> double;                  // NOLINT
    static auto UUID(int device_id) -> std::string;                   // NOLINT
    static auto PCIeBusID(int device_id) -> std::string;              // NOLINT
    static auto Name(int) -> std::string;                             // NOLINT
    static auto AccessibleDeviceIndexes() -> std::set<unsigned int>;  // NOLINT
    static auto AccessibleDevices() -> std::size_t;                   // NOLINT
};

}  // namespace mrc::internal::system

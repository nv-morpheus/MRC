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
#include "mrc/protos/architect.pb.h"
#include "mrc/utils/bytes_to_string.hpp"

#include <cstdint>
#include <ostream>
#include <string>

namespace mrc::internal::system {

/**
 * @brief GpuInfo describes an NVIDIA GPU within the hwloc topology
 *
 * This object is constructed an owned by Topology and gets serialized and deserialzied with respect to the hwloc
 * topology. The information in this object can not be extracted from the hwloc serialized topology alone, it combines
 * the runtime details from NVML. This allows us to move both topology and gpu information to the architect service.
 *
 * TODO(someone) - probably move this to a child class of topology.
 */
class GpuInfo
{
  public:
    [[nodiscard]] const CpuSet& cpu_set() const;

    [[nodiscard]] const std::string& name() const;
    [[nodiscard]] const std::string& uuid() const;
    [[nodiscard]] const std::string& cpustr() const;
    [[nodiscard]] const std::uint64_t& memory_capacity() const;
    [[nodiscard]] const std::string& pcie_bus_id() const;
    [[nodiscard]] int cuda_device_id() const;

    protos::GpuInfo serialize() const;
    static GpuInfo deserialize(const protos::GpuInfo&);

  private:
    CpuSet m_cpu_set;

    std::string m_name;
    std::string m_uuid;
    std::string m_pcie_bus_id;

    std::string m_cpustr;
    std::uint64_t m_memory_capacity;

    int m_cuda_device_id;

    // std::uint32_t m_compute_capability_major;
    // std::uint32_t m_compute_capability_minor;

    friend class Topology;
    friend std::ostream& operator<<(std::ostream& os, const GpuInfo& info)
    {
        os << "[" << info.name() << "; " << bytes_to_string(info.memory_capacity()) << "; cpu_set: " << info.cpustr()
           << "; pcie_bus_id: " << info.pcie_bus_id() << "; cuda_device_id: " << info.cuda_device_id() << "]";
        return os;
    }
};

}  // namespace mrc::internal::system

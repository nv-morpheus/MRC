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

#include "mrc/core/bitmap.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/protos/architect.pb.h"
#include "mrc/utils/macros.hpp"

#include <glog/logging.h>
#include <hwloc.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define CHECK_HWLOC(hwloc_call) \
    {                           \
        auto rc = hwloc_call;   \
        CHECK_NE(rc, -1);       \
    }

namespace mrc::internal::system {

class Topology final
{
    Topology(hwloc_topology_t, CpuSet&&, std::map<int, GpuInfo>&&);

  public:
    static std::shared_ptr<Topology> Create();                                // NOLINT
    static std::shared_ptr<Topology> Create(const TopologyOptions& options);  // NOLINT
    static std::shared_ptr<Topology> Create(const protos::Topology& msg);     // NOLINT
    static std::shared_ptr<Topology> Create(const TopologyOptions& options,   // NOLINT
                                            const protos::Topology& msg);     // NOLINT

    static std::shared_ptr<Topology> Create(const TopologyOptions& options,    // NOLINT
                                            hwloc_topology_t system_topology,  // NOLINT
                                            Bitmap cpu_set,
                                            std::map<int, GpuInfo>);

    static std::pair<hwloc_topology_t, std::map<int, GpuInfo>> Deserialize(const protos::Topology& msg);  // NOLINT
    virtual ~Topology();

    DELETE_COPYABILITY(Topology);
    DELETE_MOVEABILITY(Topology);

    /**
     * @brief CpuSet of the requested topology
     *
     * @return const CpuSet&
     */
    const CpuSet& cpu_set() const;

    /**
     * @brief CpuSet for NUMA node id in the requested topology
     *
     * @param id
     * @return const CpuSet&
     */
    const CpuSet& numa_cpuset(std::uint32_t id) const;

    /**
     * @brief number of cores in the requested topology
     *
     * @return std::uint32_t
     */
    std::uint32_t core_count() const;

    /**
     * @brief number of cpus in the requested topology
     *
     * @return std::uint32_t
     */
    std::uint32_t cpu_count() const;

    /**
     * @brief number of numa nodes / numa domain in the requested topology
     *
     * @return std::uint32_t
     */
    std::uint32_t numa_count() const;

    /**
     * @brief number of gpus and detailed gpu information
     *
     * @return std::uint32_t
     */
    std::uint32_t gpu_count() const;

    /**
     * @brief CUDA devices that match the requested options
     *
     * @param i
     * @return const GpuInfo&
     */
    const std::map<int, GpuInfo>& gpu_info() const;

    /**
     * @brief hwloc topology handle
     *
     * @return hwloc_topology_t
     */
    hwloc_topology_t handle() const;

    /**
     * @brief export the xml representation of the topology
     *
     * @return std::string
     */
    std::string export_xml() const;

    /**
     * @brief depth for object type
     *
     * @return int
     */
    [[nodiscard]] int depth_for_object(hwloc_obj_type_t) const;

    /**
     * @brief returns the i-th object at the specified depth
     *
     * @param depth
     * @param id
     * @return hwloc_obj_t
     */
    [[nodiscard]] hwloc_obj_t object_at_depth(int depth, int id) const;

    /**
     * @brief number of objects at the specified depth
     *
     * @param depth
     * @return std::uint32_t
     */
    [[nodiscard]] std::uint32_t object_count_at_depth(int depth) const;

    /**
     * @brief CpuSet for the ith object at the specified depth
     *
     * @param depth
     * @param id
     * @return CpuSet
     */
    [[nodiscard]] CpuSet cpuset_for_object(int depth, int id) const;

    /**
     * @brief NumaSet for the ith object at the specified depth
     *
     * @param depth
     * @param id
     * @return NumaSet
     */
    [[nodiscard]] NumaSet numaset_for_object(int depth, int id) const;

    /**
     * @brief NumaSet for CpuSet
     *
     * @param cpu_set
     * @return NumaSet
     */
    [[nodiscard]] NumaSet numaset_for_cpuset(const CpuSet& cpu_set) const;

    protos::Topology serialize() const;

    static void serialize_to_file(std::string path);
    static protos::Topology deserialize_from_file(std::string path);

    /**
     * @brief Determines if the CpuSet is a subset of Topology::cpu_set()
     */
    bool contains(const CpuSet& cpu_set) const;

  protected:
    [[nodiscard]] static hwloc_obj_t object_at_depth(hwloc_topology_t, int depth, int id);
    [[nodiscard]] static CpuSet cpuset_for_object(hwloc_topology_t, int depth, int id);
    Bitmap& topo_bitset();

  private:
    hwloc_topology_t m_topology;
    CpuSet m_topo_cpuset;
    int m_depth_core;
    int m_depth_cpu;
    int m_depth_numa;
    std::vector<CpuSet> m_numa_cpusets;
    std::map<int, GpuInfo> m_gpu_info;
};

}  // namespace mrc::internal::system

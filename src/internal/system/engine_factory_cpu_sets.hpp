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
#include "mrc/options/options.hpp"

#include <cstddef>
#include <map>
#include <string>

namespace mrc::internal::system {

class Topology;

struct EngineFactoryCpuSets
{
    bool is_resuable(const std::string& name) const;
    std::size_t main_cpu_id() const;

    std::map<std::string, Bitmap> fiber_cpu_sets;
    std::map<std::string, Bitmap> thread_cpu_sets;
    std::map<std::string, bool> reusable;
    Bitmap shared_cpus_set;
    bool shared_cpus_has_fibers{false};
};

/**
 * @brief Generate CpuSets for each EngineGroup
 *
 * First determine the required number of logical cpus and counts for non-overlapping types and a count for the shared
 * set. Next, reserve the logical cpus for the default group, then for each non-overlapping group and finally the
 * remaining logical cpus are reserved for the shared set.
 *
 * @param options
 * @param cpu_set
 * @return LaunchControlPlacementCpuSets
 */
extern EngineFactoryCpuSets generate_engine_factory_cpu_sets(const Topology& topology,
                                                             const Options& options,
                                                             const CpuSet& cpu_set);

}  // namespace mrc::internal::system

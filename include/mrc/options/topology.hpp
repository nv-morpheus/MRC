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

#include <string>

namespace mrc {

class TopologyOptions
{
    // use process cpu_set to restrict topology: default = true
    // user cpu_set to further restrict cpu_set : default = none / empty
    // restrict numa domains to the cpu_set    : default = true
    // options not implemented
    // restrict gpus to the cpu_set            : default = false; true not implemented
    // restrict nics to the cpu_set            : default = ????? - not sure
  public:
    TopologyOptions() = default;

    /**
     * @brief respect the process affinity as launch by the system (default: true)
     **/
    TopologyOptions& use_process_cpuset(bool default_true);

    /**
     * @brief user specified subset of logcial cpus on which to run
     **/
    TopologyOptions& user_cpuset(CpuSet&& cpu_set);

    /**
     * @brief user specified subset of logical cpu on which to run
     * @param [in] cpustr example "0-3,8-11"
     **/
    TopologyOptions& user_cpuset(std::string cpustr);

    /**
     * @brief limit the allowed numa domains to only those in the topology cpu_set (default: true)
     **/
    TopologyOptions& restrict_numa_domains(bool yn);

    /**
     * @brief limit the allowed gpus to only those near the topology cpu_set (default: false)
     **/
    TopologyOptions& restrict_gpus(bool default_false);

    /**
     * @brief ignore the display card on dgx stations (default: true)
     */
    TopologyOptions& ignore_dgx_display(bool default_true);

    [[nodiscard]] bool use_process_cpuset() const;
    [[nodiscard]] bool restrict_numa_domains() const;
    [[nodiscard]] bool restrict_gpus() const;
    [[nodiscard]] bool ignore_dgx_display() const;
    [[nodiscard]] const CpuSet& user_cpuset() const;

  private:
    bool m_use_process_cpuset{true};
    bool m_restrict_numa_domains{true};
    bool m_restrict_gpus{false};
    bool m_ignore_dgx_display{true};
    CpuSet m_user_cpuset;
};

}  // namespace mrc

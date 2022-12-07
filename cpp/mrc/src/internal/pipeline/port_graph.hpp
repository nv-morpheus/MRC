/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <map>
#include <set>
#include <string>

/**
 * @brief Graph of Pipeline Ports
 */

namespace mrc::internal::pipeline {

class Pipeline;

struct PortConnections
{
    // set of segments that have ingress ports
    std::set<std::string> ingress_segments;
    std::set<std::string> egress_segments;
};

using PortMap = std::map<std::string, PortConnections>;  // NOLINT

class PortGraph
{
  public:
    PortGraph(const Pipeline& pipeline);

    const PortMap& port_map() const;

    const std::set<std::string>& segments_with_no_ports() const;
    const std::set<std::string>& segments_with_only_ingress_ports() const;
    const std::set<std::string>& segments_with_only_egress_ports() const;

  private:
    PortMap m_port_map;

    // segments with no ports
    std::set<std::string> m_standalone;

    // segments with only ingress ports
    std::set<std::string> m_sinks;

    // segments with only egress ports
    std::set<std::string> m_sources;
};

}  // namespace mrc::internal::pipeline

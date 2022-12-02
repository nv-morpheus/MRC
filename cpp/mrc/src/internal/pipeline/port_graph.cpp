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

#include "internal/pipeline/port_graph.hpp"

#include "internal/pipeline/pipeline.hpp"
#include "internal/segment/definition.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace mrc::internal::pipeline {

PortGraph::PortGraph(const Pipeline& pipeline)
{
    for (const auto& [seg_id, seg_definition] : pipeline.segments())
    {
        auto seg_name = seg_definition->name();

        if (seg_definition->ingress_port_names().empty() and seg_definition->egress_port_names().empty())
        {
            m_standalone.insert(seg_name);
        }
        else if (seg_definition->ingress_port_names().empty())
        {
            m_sources.insert(seg_name);
        }
        else if (seg_definition->egress_port_names().empty())
        {
            m_sinks.insert(seg_name);
        }

        for (const auto& name : seg_definition->ingress_port_names())
        {
            m_port_map[name].ingress_segments.insert(seg_name);
        }

        for (const auto& name : seg_definition->egress_port_names())
        {
            m_port_map[name].egress_segments.insert(seg_name);
        }
    }
}

const PortMap& PortGraph::port_map() const
{
    return m_port_map;
}
const std::set<std::string>& PortGraph::segments_with_no_ports() const
{
    return m_standalone;
}
const std::set<std::string>& PortGraph::segments_with_only_ingress_ports() const
{
    return m_sources;
}
const std::set<std::string>& PortGraph::segments_with_only_egress_ports() const
{
    return m_sinks;
}
}  // namespace mrc::internal::pipeline

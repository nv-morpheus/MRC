/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/pipeline/pipeline_definition.hpp"

#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/segment/segment.hpp"
#include "mrc/types.hpp"

#include <glog/logging.h>

#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace mrc::pipeline {

PipelineDefinition::~PipelineDefinition() = default;

std::shared_ptr<const segment::ISegment> PipelineDefinition::register_segment(
    std::shared_ptr<const segment::ISegment> segment)
{
    auto id = m_segment_hasher.register_name(segment->name());

    // check to ensure segment is not already registered
    auto search = m_segments.find(id);
    if (search != m_segments.end())
    {
        LOG(ERROR) << "segment: " << segment->name() << " is already registered";
        throw exceptions::MrcRuntimeError("duplicate segment registration");
    }

    // check for name collisions
    for (auto& name : segment->ingress_port_names())
    {
        auto pid = m_port_hasher.register_name(name);
        DVLOG(10) << "segment: " << segment->name() << " [" << id << "] - ingress port " << name << " [" << pid << "]";
    }
    for (auto& name : segment->egress_port_names())
    {
        auto pid = m_port_hasher.register_name(name);
        DVLOG(10) << "segment: " << segment->name() << " [" << id << "] - egress port " << name << " [" << pid << "]";
    }

    const auto& [inserted_iterator, was_inserted] = m_segments.emplace(id, std::move(segment));

    return inserted_iterator->second;
}

std::shared_ptr<const segment::ISegment> PipelineDefinition::make_segment(
    const std::string& segment_name,
    segment::segment_initializer_fn_t segment_initializer)
{
    auto segdef = Segment::create(segment_name, segment_initializer);
    return this->register_segment(std::move(segdef));
}

std::shared_ptr<const segment::ISegment> PipelineDefinition::make_segment(
    const std::string& segment_name,
    segment::IngressPortsBase ingress_ports,
    segment::EgressPortsBase egress_ports,
    segment::segment_initializer_fn_t segment_initializer)
{
    auto segdef = Segment::create(segment_name, ingress_ports, egress_ports, segment_initializer);
    return this->register_segment(std::move(segdef));
}

std::shared_ptr<const segment::ISegment> PipelineDefinition::make_segment(
    const std::string& segment_name,
    segment::IngressPortsBase ingress_ports,
    segment::segment_initializer_fn_t segment_initializer)
{
    auto segdef = Segment::create(segment_name, ingress_ports, segment_initializer);
    return this->register_segment(std::move(segdef));
}

std::shared_ptr<const segment::ISegment> PipelineDefinition::make_segment(
    const std::string& segment_name,
    segment::EgressPortsBase egress_ports,
    segment::segment_initializer_fn_t segment_initializer)
{
    auto segdef = Segment::create(segment_name, egress_ports, segment_initializer);
    return this->register_segment(std::move(segdef));
}

std::shared_ptr<const segment::ISegment> PipelineDefinition::find_segment(SegmentID segment_id) const
{
    auto search = m_segments.find(segment_id);
    CHECK(search != m_segments.end());
    return search->second;
}

const std::map<SegmentID, std::shared_ptr<const segment::ISegment>>& PipelineDefinition::segments() const
{
    return m_segments;
}
// std::shared_ptr<Pipeline> PipelineDefinition::unwrap(IPipeline& pipeline)
// {
//     return pipeline.m_impl;
// }

}  // namespace mrc::pipeline

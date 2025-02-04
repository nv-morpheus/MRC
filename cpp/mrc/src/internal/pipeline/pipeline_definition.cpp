/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/segment/segment_definition.hpp"

#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/pipeline/segment.hpp"
#include "mrc/segment/egress_ports.hpp"
#include "mrc/segment/ingress_ports.hpp"
#include "mrc/types.hpp"

#include <glog/logging.h>

#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace mrc::pipeline {

PipelineDefinition::~PipelineDefinition() = default;

std::shared_ptr<PipelineDefinition> PipelineDefinition::unwrap(std::shared_ptr<IPipeline> object)
{
    // Convert to the full implementation
    auto full_object = std::dynamic_pointer_cast<PipelineDefinition>(object);

    CHECK(full_object) << "Invalid cast for PipelineDefinition. Please report to the developers";

    return full_object;
}

std::shared_ptr<const ISegment> PipelineDefinition::register_segment(std::shared_ptr<const ISegment> segment)
{
    auto id = m_segment_hasher.register_name(segment->name());

    // check to ensure segment is not already registered
    auto search = m_segments.find(id);
    if (search != m_segments.end())
    {
        LOG(ERROR) << "segment: " << segment->name() << " is already registered";
        throw exceptions::MrcRuntimeError("duplicate segment registration");
    }

    auto full_segment = segment::SegmentDefinition::unwrap(std::move(segment));

    // check for name collisions
    for (auto& name : full_segment->ingress_port_names())
    {
        auto pid = m_port_hasher.register_name(name);
        DVLOG(10) << "segment: " << full_segment->name() << " [" << id << "] - ingress port " << name << " [" << pid
                  << "]";
    }
    for (auto& name : full_segment->egress_port_names())
    {
        auto pid = m_port_hasher.register_name(name);
        DVLOG(10) << "segment: " << full_segment->name() << " [" << id << "] - egress port " << name << " [" << pid
                  << "]";
    }

    const auto& [inserted_iterator, was_inserted] = m_segments.emplace(id, std::move(full_segment));

    return inserted_iterator->second;
}

std::shared_ptr<const ISegment> PipelineDefinition::make_segment(const std::string& segment_name,
                                                                 segment::segment_initializer_fn_t segment_initializer)
{
    auto segdef = Segment::create(segment_name, segment_initializer);
    return this->register_segment(std::move(segdef));
}

std::shared_ptr<const ISegment> PipelineDefinition::make_segment(const std::string& segment_name,
                                                                 segment::IngressPortsBase ingress_ports,
                                                                 segment::EgressPortsBase egress_ports,
                                                                 segment::segment_initializer_fn_t segment_initializer)
{
    auto segdef = Segment::create(segment_name, ingress_ports, egress_ports, segment_initializer);
    return this->register_segment(std::move(segdef));
}

std::shared_ptr<const ISegment> PipelineDefinition::make_segment(const std::string& segment_name,
                                                                 segment::IngressPortsBase ingress_ports,
                                                                 segment::segment_initializer_fn_t segment_initializer)
{
    auto segdef = Segment::create(segment_name, ingress_ports, segment_initializer);
    return this->register_segment(std::move(segdef));
}

std::shared_ptr<const ISegment> PipelineDefinition::make_segment(const std::string& segment_name,
                                                                 segment::EgressPortsBase egress_ports,
                                                                 segment::segment_initializer_fn_t segment_initializer)
{
    auto segdef = Segment::create(segment_name, egress_ports, segment_initializer);
    return this->register_segment(std::move(segdef));
}

std::shared_ptr<const segment::SegmentDefinition> PipelineDefinition::find_segment(SegmentID segment_id) const
{
    auto search = m_segments.find(segment_id);
    CHECK(search != m_segments.end());
    return search->second;
}

const std::map<SegmentID, std::shared_ptr<const segment::SegmentDefinition>>& PipelineDefinition::segments() const
{
    return m_segments;
}

}  // namespace mrc::pipeline

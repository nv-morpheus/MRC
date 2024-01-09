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

#include "internal/pipeline/manifold_definition.hpp"
#include "internal/segment/segment_definition.hpp"

#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/port_registry.hpp"
#include "mrc/pipeline/segment.hpp"
#include "mrc/segment/egress_ports.hpp"
#include "mrc/segment/ingress_ports.hpp"
#include "mrc/segment/ports.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>

#include <memory>
#include <ostream>
#include <string>
#include <typeindex>
#include <utility>

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
    for (const auto& [name, port_info] : full_segment->ingress_port_infos())
    {
        auto pid = m_port_hasher.register_name(name);
        DVLOG(10) << "segment: " << full_segment->name() << " [" << id << "] - ingress port " << name << " [" << pid
                  << "]";

        std::type_index type_index = port_info->type_index;

        if (!m_manifolds.contains(name))
        {
            // From the port utils, lookup the default builder function for this manifold
            auto port_util = node::PortRegistry::find_port_util(type_index);

            // Create a manifold definition from the default builder
            m_manifolds[name] = std::make_shared<ManifoldDefinition>(name, type_index, port_util->manifold_builder_fn);
        }

        // Now check that the type IDs are equal
        CHECK(m_manifolds[name]->type_index() == type_index)
            << "Mismatched types on manifold with name '" << name << "'. Existing Type: " << mrc::type_name(type_index)
            << ". Registering Type: " << mrc::type_name(type_index);
    }

    for (const auto& [name, port_info] : full_segment->egress_port_infos())
    {
        auto pid = m_port_hasher.register_name(name);
        DVLOG(10) << "segment: " << full_segment->name() << " [" << id << "] - egress port " << name << " [" << pid
                  << "]";

        std::type_index type_index = port_info->type_index;

        if (!m_manifolds.contains(name))
        {
            // From the port utils, lookup the default builder function for this manifold
            auto port_util = node::PortRegistry::find_port_util(type_index);

            // Create a manifold definition from the default builder
            m_manifolds[name] = std::make_shared<ManifoldDefinition>(name, type_index, port_util->manifold_builder_fn);
        }

        // Now check that the type IDs are equal
        CHECK(m_manifolds[name]->type_index() == type_index)
            << "Mismatched types on manifold with name '" << name << "'. Existing Type: " << mrc::type_name(type_index)
            << ". Registering Type: " << mrc::type_name(type_index);
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

std::vector<std::shared_ptr<const ISegment>> PipelineDefinition::segments() const
{
    std::vector<std::shared_ptr<const ISegment>> segments;

    for (const auto& [id, segment] : m_segments)
    {
        segments.push_back(segment);
    }

    return segments;
}

const std::map<SegmentID, std::shared_ptr<const segment::SegmentDefinition>>& PipelineDefinition::segment_defs() const
{
    return m_segments;
}

std::shared_ptr<const ManifoldDefinition> PipelineDefinition::find_manifold(const std::string& manifold_name) const
{
    auto search = m_manifolds.find(manifold_name);
    CHECK(search != m_manifolds.end()) << "Manifold with name '" << manifold_name << "' not found.";
    return search->second;
}

std::shared_ptr<const segment::SegmentDefinition> PipelineDefinition::find_segment(SegmentID segment_id) const
{
    auto search = m_segments.find(segment_id);
    CHECK(search != m_segments.end());
    return search->second;
}

}  // namespace mrc::pipeline

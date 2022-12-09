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

#include "internal/pipeline/pipeline.hpp"

#include "internal/segment/definition.hpp"

#include "mrc/engine/pipeline/ipipeline.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/types.hpp"

#include <glog/logging.h>

#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace mrc::internal::pipeline {

void Pipeline::add_segment(std::shared_ptr<const segment::Definition> segment)
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

    m_segments[id] = segment;
}

std::shared_ptr<const segment::Definition> Pipeline::find_segment(SegmentID segment_id) const
{
    auto search = m_segments.find(segment_id);
    CHECK(search != m_segments.end());
    return search->second;
}

const std::map<SegmentID, std::shared_ptr<const segment::Definition>>& Pipeline::segments() const
{
    return m_segments;
}
std::shared_ptr<Pipeline> Pipeline::unwrap(IPipeline& pipeline)
{
    return pipeline.m_impl;
}
}  // namespace mrc::internal::pipeline

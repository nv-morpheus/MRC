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

#pragma once

#include "internal/segment/segment_definition.hpp"
#include "internal/utils/collision_detector.hpp"

#include "mrc/pipeline/pipeline.hpp"
#include "mrc/types.hpp"

#include <map>
#include <memory>

namespace mrc::segment {
class ISegment;
}

namespace mrc::pipeline {

class PipelineDefinition : public IPipeline
{
  public:
    ~PipelineDefinition() override;

    static std::shared_ptr<PipelineDefinition> unwrap(std::shared_ptr<IPipeline> object);

    std::shared_ptr<const segment::ISegment> register_segment(std::shared_ptr<const segment::ISegment> segment) override;

    std::shared_ptr<const segment::ISegment> make_segment(
        const std::string& segment_name,
        segment::segment_initializer_fn_t segment_initializer) override;

    std::shared_ptr<const segment::ISegment> make_segment(
        const std::string& segment_name,
        segment::IngressPortsBase ingress_ports,
        segment::EgressPortsBase egress_ports,
        segment::segment_initializer_fn_t segment_initializer) override;

    std::shared_ptr<const segment::ISegment> make_segment(
        const std::string& segment_name,
        segment::IngressPortsBase ingress_ports,
        segment::segment_initializer_fn_t segment_initializer) override;

    std::shared_ptr<const segment::ISegment> make_segment(
        const std::string& segment_name,
        segment::EgressPortsBase egress_ports,
        segment::segment_initializer_fn_t segment_initializer) override;

    // static std::shared_ptr<PipelineDefinition> unwrap(IPipeline& pipeline);

    const std::map<SegmentID, std::shared_ptr<const segment::SegmentDefinition>>& segments() const;

    std::shared_ptr<const segment::SegmentDefinition> find_segment(SegmentID segment_id) const;

  private:
    utils::CollisionDetector m_segment_hasher;
    utils::CollisionDetector m_port_hasher;

    std::map<SegmentID, std::shared_ptr<const segment::SegmentDefinition>> m_segments;
};

}  // namespace mrc::pipeline

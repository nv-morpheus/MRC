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

#include "mrc/segment/initializers.hpp"
#include "mrc/utils/macros.hpp"

#include <memory>
#include <string>
#include <utility>

namespace mrc::segment {
struct EgressPortsBase;
struct IngressPortsBase;
class ISegment;
}  // namespace mrc::segment

// work-around for known iwyu issue
// https://github.com/include-what-you-use/include-what-you-use/issues/908
// IWYU pragma: no_include <algorithm>

namespace mrc::pipeline {

class IPipeline
{
  public:
    IPipeline()          = default;
    virtual ~IPipeline() = default;

    DELETE_COPYABILITY(IPipeline);

    /**
     * @brief register a segment
     * @param [in] segment
     * @throws
     **/
    virtual std::shared_ptr<const segment::ISegment> register_segment(
        std::shared_ptr<const segment::ISegment> segment) = 0;

    /**
     * @brief register multiple segments
     *
     * @tparam SegmentDefs
     * @param segment_defs
     */
    template <typename... SegmentDefsT>
    std::vector<std::shared_ptr<const segment::ISegment>> register_segments(SegmentDefsT&&... segment_defs)
    {
        auto segments = std::vector<std::shared_ptr<const segment::ISegment>>{
            {this->register_segment(std::forward<SegmentDefsT>(segment_defs))...}};

        return segments;
    }

    /**
     * Create a segment definition, which describes how to create new Segment instances.
     * @tparam InputTypes Segment ingress interface datatypes
     * @tparam OutputTypes Segment egress interface datatypes
     * @param p Parent pipeline
     * @param segment_name Unique name to assign to segments built from this definition
     * @param segment_initializer User defined lambda function which will be used to initialize
     *  new segments.
     * @return A shared pointer to a new segment::ISegment
     */
    virtual std::shared_ptr<const segment::ISegment> make_segment(
        const std::string& segment_name,
        segment::segment_initializer_fn_t segment_initializer) = 0;

    /**
     * Create a segment definition, which describes how to create new Segment instances.
     * @tparam InputTypes Segment ingress interface datatypes
     * @tparam OutputTypes Segment egress interface datatypes
     * @param p Parent pipeline
     * @param ingress_ports Porttypes object describing a Segment's ingress ports
     * @param egress_ports Porttypes object describing a Segment's egress ports
     * @param segment_name Unique name to assign to segments built from this definition
     * @param segment_initializer User defined lambda function which will be used to initialize
     *  new segments.
     * @return A shared pointer to a new segment::ISegment
     */
    virtual std::shared_ptr<const segment::ISegment> make_segment(
        const std::string& segment_name,
        segment::IngressPortsBase ingress_ports,
        segment::EgressPortsBase egress_ports,
        segment::segment_initializer_fn_t segment_initializer) = 0;

    /**
     * Create a segment definition, which describes how to create new Segment instances.
     * @tparam InputTypes Segment ingress interface datatypes
     * @tparam OutputTypes Segment egress interface datatypes
     * @param p Parent pipeline
     * @param ingress_ports Porttypes object describing a Segment's ingress ports
     * @param egress_ports Porttypes object describing a Segment's egress ports
     * @param segment_name Unique name to assign to segments built from this definition
     * @param segment_initializer User defined lambda function which will be used to initialize
     *  new segments.
     * @return A shared pointer to a new segment::ISegment
     */
    virtual std::shared_ptr<const segment::ISegment> make_segment(
        const std::string& segment_name,
        segment::IngressPortsBase ingress_ports,
        segment::segment_initializer_fn_t segment_initializer) = 0;

    /**
     * Create a segment definition, which describes how to create new Segment instances.
     * @tparam InputTypes Segment ingress interface datatypes
     * @tparam OutputTypes Segment egress interface datatypes
     * @param p Parent pipeline
     * @param ingress_ports Porttypes object describing a Segment's ingress ports
     * @param egress_ports Porttypes object describing a Segment's egress ports
     * @param segment_name Unique name to assign to segments built from this definition
     * @param segment_initializer User defined lambda function which will be used to initialize
     *  new segments.
     * @return A shared pointer to a new segment::ISegment
     */
    virtual std::shared_ptr<const segment::ISegment> make_segment(
        const std::string& segment_name,
        segment::EgressPortsBase egress_ports,
        segment::segment_initializer_fn_t segment_initializer) = 0;
};

}  // namespace mrc::pipeline

namespace mrc {
std::unique_ptr<pipeline::IPipeline> make_pipeline();
}

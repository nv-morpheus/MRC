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

#include "mrc/engine/pipeline/ipipeline.hpp"
#include "mrc/segment/initializers.hpp"
#include "mrc/utils/macros.hpp"

#include <memory>
#include <string>
#include <utility>

namespace mrc::segment {
struct EgressPortsBase;
struct IngressPortsBase;
class Definition;
}  // namespace mrc::segment

// work-around for known iwyu issue
// https://github.com/include-what-you-use/include-what-you-use/issues/908
// IWYU pragma: no_include <algorithm>

namespace mrc::pipeline {

class Pipeline final : public internal::pipeline::IPipeline
{
    Pipeline()   = default;
    using base_t = internal::pipeline::IPipeline;

  public:
    static std::unique_ptr<Pipeline> create();

    ~Pipeline() final = default;

    DELETE_COPYABILITY(Pipeline);
    DELETE_MOVEABILITY(Pipeline);

    /**
     * @brief register a segment
     * @param [in] segment
     * @throws
     **/
    void register_segment(std::shared_ptr<segment::Definition> segment);

    /**
     * @brief register multiple segments
     *
     * @tparam SegmentDefs
     * @param segment_defs
     */
    template <typename... SegmentDefs>  // NOLINT
    void register_segments(SegmentDefs&&... segment_defs)
    {
        (register_segment(std::forward<SegmentDefs>(segment_defs)), ...);
    }

    /**
     * Create a segment definition, which describes how to create new Segment instances.
     * @tparam InputTypes Segment ingress interface datatypes
     * @tparam OutputTypes Segment egress interface datatypes
     * @param p Parent pipeline
     * @param segment_name Unique name to assign to segments built from this definition
     * @param segment_initializer User defined lambda function which will be used to initialize
     *  new segments.
     * @return A shared pointer to a new segment::Definition
     */
    std::shared_ptr<segment::Definition> make_segment(const std::string& segment_name,
                                                      segment::segment_initializer_fn_t segment_initializer);

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
     * @return A shared pointer to a new segment::Definition
     */
    std::shared_ptr<segment::Definition> make_segment(const std::string& segment_name,
                                                      segment::IngressPortsBase ingress_ports,
                                                      segment::EgressPortsBase egress_ports,
                                                      segment::segment_initializer_fn_t segment_initializer);

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
     * @return A shared pointer to a new segment::Definition
     */
    std::shared_ptr<segment::Definition> make_segment(const std::string& segment_name,
                                                      segment::IngressPortsBase ingress_ports,
                                                      segment::segment_initializer_fn_t segment_initializer);

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
     * @return A shared pointer to a new segment::Definition
     */
    std::shared_ptr<segment::Definition> make_segment(const std::string& segment_name,
                                                      segment::EgressPortsBase egress_ports,
                                                      segment::segment_initializer_fn_t segment_initializer);
};

std::unique_ptr<Pipeline> make_pipeline();

}  // namespace mrc::pipeline

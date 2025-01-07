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

#pragma once

#include "mrc/segment/initializers.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/macros.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mrc::segment {
struct EgressPortsBase;
struct IngressPortsBase;
}  // namespace mrc::segment

namespace mrc::pipeline {

class ISegment
{
  public:
    virtual ~ISegment() = default;

    DELETE_COPYABILITY(ISegment);

    virtual SegmentID id() const                                = 0;
    virtual const std::string& name() const                     = 0;
    virtual std::vector<std::string> ingress_port_names() const = 0;
    virtual std::vector<std::string> egress_port_names() const  = 0;

  protected:
    ISegment() = default;
};
}  // namespace mrc::pipeline

namespace mrc {

// This helper class if for backwards compatibility only
class Segment final
{
  public:
    static std::unique_ptr<const pipeline::ISegment> create(std::string name,
                                                            segment::IngressPortsBase ingress_ports,
                                                            segment::EgressPortsBase egress_ports,
                                                            segment::segment_initializer_fn_t initializer);

    static std::unique_ptr<const pipeline::ISegment> create(std::string name,
                                                            segment::EgressPortsBase egress_ports,
                                                            segment::segment_initializer_fn_t initializer);

    static std::unique_ptr<const pipeline::ISegment> create(std::string name,
                                                            segment::IngressPortsBase ingress_ports,
                                                            segment::segment_initializer_fn_t initializer);

    static std::unique_ptr<const pipeline::ISegment> create(std::string name,
                                                            segment::segment_initializer_fn_t initializer);
};

std::unique_ptr<const pipeline::ISegment> make_segment(std::string name,
                                                       segment::IngressPortsBase ingress_ports,
                                                       segment::EgressPortsBase egress_ports,
                                                       segment::segment_initializer_fn_t initializer);

std::unique_ptr<const pipeline::ISegment> make_segment(std::string name,
                                                       segment::EgressPortsBase egress_ports,
                                                       segment::segment_initializer_fn_t initializer);

std::unique_ptr<const pipeline::ISegment> make_segment(std::string name,
                                                       segment::IngressPortsBase ingress_ports,
                                                       segment::segment_initializer_fn_t initializer);

std::unique_ptr<const pipeline::ISegment> make_segment(std::string name, segment::segment_initializer_fn_t initializer);

}  // namespace mrc

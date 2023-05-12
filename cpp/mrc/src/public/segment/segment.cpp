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

#include "mrc/segment/segment.hpp"

#include "internal/segment/segment_definition.hpp"

#include <memory>

namespace mrc::segment {
class IBuilder;
}

namespace mrc {

std::unique_ptr<const segment::ISegment> Segment::create(std::string name,
                                                         segment::IngressPortsBase ingress_ports,
                                                         segment::EgressPortsBase egress_ports,
                                                         segment::segment_initializer_fn_t initializer)
{
    return std::make_unique<segment::SegmentDefinition>(std::move(name),
                                                        std::move(ingress_ports),
                                                        std::move(egress_ports),
                                                        std::move(initializer));
}

std::unique_ptr<const segment::ISegment> Segment::create(std::string name,
                                                         segment::EgressPortsBase egress_ports,
                                                         segment::segment_initializer_fn_t initializer)
{
    return Segment::create(std::move(name), {}, std::move(egress_ports), std::move(initializer));
}

std::unique_ptr<const segment::ISegment> Segment::create(std::string name,
                                                         segment::IngressPortsBase ingress_ports,
                                                         segment::segment_initializer_fn_t initializer)
{
    return Segment::create(std::move(name), std::move(ingress_ports), {}, std::move(initializer));
}

std::unique_ptr<const segment::ISegment> Segment::create(std::string name,
                                                         segment::segment_initializer_fn_t initializer)
{
    return Segment::create(std::move(name), {}, {}, std::move(initializer));
}

std::unique_ptr<const segment::ISegment> make_segment(std::string name,
                                                      segment::IngressPortsBase ingress_ports,
                                                      segment::EgressPortsBase egress_ports,
                                                      segment::segment_initializer_fn_t initializer)
{
    return Segment::create(std::move(name), std::move(ingress_ports), std::move(egress_ports), std::move(initializer));
}

std::unique_ptr<const segment::ISegment> make_segment(std::string name,
                                                      segment::EgressPortsBase egress_ports,
                                                      segment::segment_initializer_fn_t initializer)
{
    return Segment::create(std::move(name), {}, std::move(egress_ports), std::move(initializer));
}

std::unique_ptr<const segment::ISegment> make_segment(std::string name,
                                                      segment::IngressPortsBase ingress_ports,
                                                      segment::segment_initializer_fn_t initializer)
{
    return Segment::create(std::move(name), std::move(ingress_ports), {}, std::move(initializer));
}

std::unique_ptr<const segment::ISegment> make_segment(std::string name, segment::segment_initializer_fn_t initializer)
{
    return Segment::create(std::move(name), {}, {}, std::move(initializer));
}

}  // namespace mrc

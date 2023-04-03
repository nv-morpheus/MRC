/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/pipeline/pipeline.hpp"

#include "mrc/segment/definition.hpp"
#include "mrc/segment/egress_ports.hpp"
#include "mrc/segment/ingress_ports.hpp"

namespace mrc::pipeline {

std::unique_ptr<Pipeline> Pipeline::create()
{
    return std::unique_ptr<Pipeline>(new Pipeline());
}

void Pipeline::register_segment(std::shared_ptr<segment::Definition> segment)
{
    base_t::register_segment(std::move(segment));
}

std::shared_ptr<segment::Definition> Pipeline::make_segment(const std::string& segment_name,
                                                            segment::segment_initializer_fn_t segment_initializer)
{
    auto segdef = segment::Definition::create(segment_name, segment_initializer);
    this->register_segment(segdef);
    return segdef;
};

std::shared_ptr<segment::Definition> Pipeline::make_segment(const std::string& segment_name,
                                                            segment::IngressPortsBase ingress_ports,
                                                            segment::EgressPortsBase egress_ports,
                                                            segment::segment_initializer_fn_t segment_initializer)
{
    auto segdef = segment::Definition::create(segment_name, ingress_ports, egress_ports, segment_initializer);
    this->register_segment(segdef);
    return segdef;
};

std::shared_ptr<segment::Definition> Pipeline::make_segment(const std::string& segment_name,
                                                            segment::IngressPortsBase ingress_ports,
                                                            segment::segment_initializer_fn_t segment_initializer)
{
    auto segdef = segment::Definition::create(segment_name, ingress_ports, segment_initializer);
    this->register_segment(segdef);
    return segdef;
};

std::shared_ptr<segment::Definition> Pipeline::make_segment(const std::string& segment_name,
                                                            segment::EgressPortsBase egress_ports,
                                                            segment::segment_initializer_fn_t segment_initializer)
{
    auto segdef = segment::Definition::create(segment_name, egress_ports, segment_initializer);
    this->register_segment(segdef);
    return segdef;
};

std::unique_ptr<Pipeline> make_pipeline()
{
    return Pipeline::create();
}

}  // namespace mrc::pipeline

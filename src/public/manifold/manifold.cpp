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

#include "mrc/manifold/manifold.hpp"

#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/pipeline/resources.hpp"
#include "mrc/segment/utils.hpp"
#include "mrc/types.hpp"

#include <glog/logging.h>

#include <ostream>
#include <string>
#include <utility>

namespace mrc::manifold {

Manifold::Manifold(PortName port_name, pipeline::Resources& resources) :
  m_port_name(std::move(port_name)),
  m_resources(resources)
{}

const PortName& Manifold::port_name() const
{
    return m_port_name;
}

pipeline::Resources& Manifold::resources()
{
    return m_resources;
}

void Manifold::add_input(const SegmentAddress& address, node::SourcePropertiesBase* input_source)
{
    DVLOG(3) << "manifold " << this->port_name() << ": connecting to upstream segment " << segment::info(address);
    do_add_input(address, input_source);
    DVLOG(10) << "manifold " << this->port_name() << ": completed connection to upstream segment "
              << segment::info(address);
}

void Manifold::add_output(const SegmentAddress& address, node::SinkPropertiesBase* output_sink)
{
    DVLOG(3) << "manifold " << this->port_name() << ": connecting to downstream segment " << segment::info(address);
    do_add_output(address, output_sink);
    DVLOG(10) << "manifold " << this->port_name() << ": completed connection to downstream segment "
              << segment::info(address);
}

}  // namespace mrc::manifold

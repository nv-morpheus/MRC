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

#include "mrc/manifold/manifold.hpp"

#include "mrc/segment/utils.hpp"
#include "mrc/types.hpp"

#include <glog/logging.h>

#include <ostream>
#include <string>
#include <utility>

namespace mrc::manifold {

Manifold::Manifold(PortName port_name, runnable::IRunnableResources& resources) :
  m_port_name(std::move(port_name)),
  m_resources(resources)
{}

Manifold::~Manifold() = default;

const PortName& Manifold::port_name() const
{
    return m_port_name;
}

runnable::IRunnableResources& Manifold::resources()
{
    return m_resources;
}

const std::string& Manifold::info() const
{
    return m_info;
}

void Manifold::add_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source)
{
    DVLOG(3) << "manifold " << this->port_name() << ": connecting to upstream segment " << segment::info(address);
    do_add_input(address, input_source);
    DVLOG(10) << "manifold " << this->port_name() << ": completed connection to upstream segment "
              << segment::info(address);
}

void Manifold::add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink)
{
    DVLOG(3) << "manifold " << this->port_name() << ": connecting to downstream segment " << segment::info(address);
    do_add_output(address, output_sink);
    DVLOG(10) << "manifold " << this->port_name() << ": completed connection to downstream segment "
              << segment::info(address);
}

}  // namespace mrc::manifold

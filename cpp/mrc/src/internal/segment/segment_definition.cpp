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

#include "internal/segment/segment_definition.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/segment/egress_ports.hpp"
#include "mrc/segment/ingress_ports.hpp"
#include "mrc/types.hpp"

#include <glog/logging.h>

#include <cstdint>
#include <ostream>
#include <set>
#include <utility>

namespace mrc::segment {

SegmentDefinition::SegmentDefinition(std::string name,
                                     IngressPortsBase ingress_ports,
                                     EgressPortsBase egress_ports,
                                     segment_initializer_fn_t initializer) :
  m_id(segment_name_hash(name)),
  m_name(std::move(name)),
  m_ingress_initializers(ingress_ports.get_initializers()),
  m_egress_initializers(egress_ports.get_initializers()),
  m_initializer_fn(std::move(initializer))
{
    validate_ports();
}

std::shared_ptr<const SegmentDefinition> SegmentDefinition::unwrap(std::shared_ptr<const pipeline::ISegment> object)
{
    // Convert to the full implementation
    auto full_object = std::dynamic_pointer_cast<const SegmentDefinition>(object);

    CHECK(full_object) << "Invalid cast for SegmentDefinition. Please report to the developers";

    return full_object;
}

SegmentID SegmentDefinition::id() const
{
    return m_id;
}

const std::string& SegmentDefinition::name() const
{
    return m_name;
}

std::vector<std::string> SegmentDefinition::ingress_port_names() const
{
    std::vector<std::string> names;
    for (const auto& [name, init] : m_ingress_initializers)
    {
        names.push_back(name);
    }
    return names;
}
std::vector<std::string> SegmentDefinition::egress_port_names() const
{
    std::vector<std::string> names;
    for (const auto& [name, init] : m_egress_initializers)
    {
        names.push_back(name);
    }
    return names;
}

const segment_initializer_fn_t& SegmentDefinition::initializer_fn() const
{
    return m_initializer_fn;
}

const std::map<std::string, ::mrc::segment::egress_initializer_t>& SegmentDefinition::egress_initializers() const
{
    return m_egress_initializers;
}

const std::map<std::string, ::mrc::segment::ingress_initializer_t>& SegmentDefinition::ingress_initializers() const
{
    return m_ingress_initializers;
}

void SegmentDefinition::validate_ports() const
{
    std::vector<std::string> names;

    for (const auto& [name, initializer] : m_ingress_initializers)
    {
        names.push_back(name);
    }
    for (const auto& [name, initializer] : m_egress_initializers)
    {
        names.push_back(name);
    }

    // check for uniqueness in port names
    std::set<std::string> port_names(names.begin(), names.end());
    if (port_names.size() != names.size())
    {
        // LOG(ERROR) << info() << "ingress and egress port names must be unique";
        throw exceptions::MrcRuntimeError("ingress and egress port names must be unique");
    }

    // check for hash collision over all port names
    std::set<std::uint16_t> port_hashes;
    for (const auto& name : names)
    {
        port_hashes.insert(port_name_hash(name));
    }
    if (port_hashes.size() != names.size())
    {
        // todo(ryan) - improve logging - print out each name and hash
        // LOG(ERROR) << info() << " hash collision detected on port names";
        throw exceptions::MrcRuntimeError("hash collection detected in port names");
    }
}

}  // namespace mrc::segment

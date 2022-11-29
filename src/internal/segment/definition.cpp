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

#include "internal/segment/definition.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/engine/segment/idefinition.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/types.hpp"

#include <cstdint>
#include <set>
#include <utility>

namespace mrc::internal::segment {

Definition::Definition(std::string name,
                       std::map<std::string, IDefinition::ingress_initializer_t> ingress_initializers,
                       std::map<std::string, IDefinition::egress_initializer_t> egress_initializers,
                       IDefinition::backend_initializer_fn_t backend_initializer) :
  m_name(name),
  m_id(segment_name_hash(name)),
  m_backend_initializer(std::move(backend_initializer)),
  m_ingress_initializers(std::move(ingress_initializers)),
  m_egress_initializers(std::move(egress_initializers))
{
    validate_ports();
}

const std::string& Definition::name() const
{
    return m_name;
}

SegmentID Definition::id() const
{
    return m_id;
}

std::vector<std::string> Definition::ingress_port_names() const
{
    std::vector<std::string> names;
    for (const auto& [name, init] : m_ingress_initializers)
    {
        names.push_back(name);
    }
    return names;
}
std::vector<std::string> Definition::egress_port_names() const
{
    std::vector<std::string> names;
    for (const auto& [name, init] : m_egress_initializers)
    {
        names.push_back(name);
    }
    return names;
}

// namespace mrc::internal::segment {
// Definition::Definition(std::string name,
//                        std::map<std::string, ingress_initializer_t> ingress_initializers,
//                        std::map<std::string, egress_initializer_t> egress_initializers,
//                        backend_initializer_fn_t initializer) :
//   m_name(name),
//   m_id(segment_name_hash(name)),
//   m_initializer(std::move(initializer)),
//   m_ingress_initializers(std::move(ingress_initializers)),
//   m_egress_initializers(std::move(egress_initializers))
// {
//     validate_ports();
// }
// const std::string& Definition::name() const
// {
//     return m_name;
// }
// const SegmentID& Definition::id() const
// {
//     return m_id;
// }
// std::vector<std::string> Definition::ingress_port_names() const
// {
//     std::vector<std::string> names;
//     for (const auto& [name, init] : m_ingress_initializers)
//     {
//         names.push_back(name);
//     }
//     return names;
// }
// std::vector<std::string> Definition::egress_port_names() const
// {
//     std::vector<std::string> names;
//     for (const auto& [name, init] : m_egress_initializers)
//     {
//         names.push_back(name);
//     }
//     return names;
// }
// std::string Definition::info() const
// {
//     std::stringstream ss;
//     ss << "[Segment::Definition " << name() << "/" << id() << "]";
//     return ss.str();
// }
// std::string Definition::info(SegmentRank rank) const
// {
//     std::stringstream ss;
//     auto address = segment_address_encode(id(), rank);
//     ss << "[Segment: " << name() << "; " << id() << "/" << rank << "/" << address << "]";
//     return ss.str();
// }
void Definition::validate_ports() const
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

// protos::SegmentDefinition Definition::serialize() const
// {
//     protos::SegmentDefinition segment;

//     // name and id
//     segment.set_name(this->name());
//     segment.set_id(this->id());

//     // ingress ports
//     for (const auto& ingress_name : this->ingress_port_names())
//     {
//         auto* ingress = segment.add_ingress_ports();
//         ingress->set_name(ingress_name);
//         ingress->set_id(port_name_hash(ingress_name));
//     }

//     // egress ports
//     for (int i = 0; i < this->egress_port_names().size(); ++i)
//     {
//         auto* egress = segment.add_egress_ports();
//         egress->set_name(this->egress_port_names()[i]);
//         egress->set_id(port_name_hash(this->egress_port_names()[i]));
//     }

//     return segment;
// }

// const DefinitionBackend::backend_initializer_fn_t& Definition::initializer_fn() const
// {
//     return m_initializer;
// }
// const std::map<std::string, DefinitionBackend::egress_initializer_t>& Definition::egress_initializers() const
// {
//     return m_egress_initializers;
// }
// const std::map<std::string, DefinitionBackend::ingress_initializer_t>& Definition::ingress_initializers() const
// {
//     return m_ingress_initializers;
// }
// }  // namespace mrc::internal::segment

}  // namespace mrc::internal::segment

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

#include "internal/segment/definition.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/types.hpp"

#include <cstdint>
#include <set>
#include <utility>

namespace mrc::segment {

Definition::Definition(std::string name,
                       std::map<std::string, ::mrc::segment::ingress_initializer_t> ingress_initializers,
                       std::map<std::string, ::mrc::segment::egress_initializer_t> egress_initializers,
                       ::mrc::segment::backend_initializer_fn_t backend_initializer) :
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

const ::mrc::segment::backend_initializer_fn_t& Definition::initializer_fn() const
{
    return m_backend_initializer;
}

const std::map<std::string, ::mrc::segment::egress_initializer_t>& Definition::egress_initializers() const
{
    return m_egress_initializers;
}

const std::map<std::string, ::mrc::segment::ingress_initializer_t>& Definition::ingress_initializers() const
{
    return m_ingress_initializers;
}

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

}  // namespace mrc::segment

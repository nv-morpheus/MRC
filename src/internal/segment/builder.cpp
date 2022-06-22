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

#include "internal/segment/builder.hpp"

#include <srf/core/addresses.hpp>
#include <srf/exceptions/runtime_error.hpp>
#include <srf/metrics/counter.hpp>
#include <srf/metrics/registry.hpp>
#include <srf/runnable/launchable.hpp>
#include <srf/segment/egress_port.hpp>
#include <srf/segment/ingress_port.hpp>

#include <glog/logging.h>

#include <ostream>
#include <utility>

namespace srf::internal::segment {

Builder::Builder(std::shared_ptr<const Definition> segdef,
                 SegmentRank rank,
                 pipeline::Resources& resources,
                 std::size_t default_partition_id) :
  m_definition(std::move(segdef)),
  m_resources(resources),
  m_default_partition_id(default_partition_id)
{
    auto address = segment_address_encode(definition().id(), rank);

    // construct ingress ports
    for (const auto& [name, initializer] : definition().ingress_initializers())
    {
        DVLOG(10) << "constructing ingress_port: " << name;
        m_ingress_ports[name] = initializer(address);
        m_objects[name]       = m_ingress_ports[name];
    }

    // construct egress ports
    for (const auto& [name, initializer] : definition().egress_initializers())
    {
        DVLOG(10) << "constructing egress_port: " << name;
        m_egress_ports[name] = initializer(address);
        m_objects[name]      = m_egress_ports[name];
    }

    definition().initializer_fn()(*this);
}

const std::string& Builder::name() const
{
    CHECK(m_definition);
    return m_definition->name();
}

bool Builder::has_object(const std::string& name) const
{
    auto search = m_objects.find(name);
    return bool(search != m_objects.end());
}

srf::segment::ObjectProperties& Builder::find_object(const std::string& name)
{
    auto search = m_objects.find(name);
    if (search == m_objects.end())
    {
        LOG(ERROR) << "Unable to find segment object with name: " << name;
        throw exceptions::SrfRuntimeError("unable to find segment object with name " + name);
    }
    return *(search->second);
}

std::shared_ptr<::srf::segment::IngressPortBase> Builder::get_ingress_base(const std::string& name)
{
    auto search = m_ingress_ports.find(name);
    if (search != m_ingress_ports.end())
    {
        return search->second;
    }
    return nullptr;
}

std::shared_ptr<::srf::segment::EgressPortBase> Builder::get_egress_base(const std::string& name)
{
    auto search = m_egress_ports.find(name);
    if (search != m_egress_ports.end())
    {
        return search->second;
    }
    return nullptr;
}

void Builder::add_object(const std::string& name, std::shared_ptr<::srf::segment::ObjectProperties> object)
{
    if (has_object(name))
    {
        LOG(ERROR) << "A Object named " << name << " is already registered";
        throw exceptions::SrfRuntimeError("duplicate name detected - name owned by a node");
    }
    m_objects[name] = std::move(object);
}

void Builder::add_runnable(const std::string& name, std::shared_ptr<runnable::Launchable> runnable)
{
    if (has_object(name))
    {
        LOG(ERROR) << "A Object named " << name << " is already registered";
        throw exceptions::SrfRuntimeError("duplicate name detected - name owned by a node");
    }
    m_nodes[name] = std::move(runnable);
}

const std::map<std::string, std::shared_ptr<::srf::segment::EgressPortBase>>& Builder::egress_ports() const
{
    return m_egress_ports;
}

const std::map<std::string, std::shared_ptr<::srf::segment::IngressPortBase>>& Builder::ingress_ports() const
{
    return m_ingress_ports;
}

const Definition& Builder::definition() const
{
    CHECK(m_definition);
    return *m_definition;
}
const std::map<std::string, std::shared_ptr<runnable::Launchable>>& Builder::nodes() const
{
    return m_nodes;
}
std::function<void(std::int64_t)> Builder::make_throughput_counter(const std::string& name)
{
    auto counter = m_resources.metrics_registry().make_throughput_counter(name);
    return [counter](std::int64_t ticks) mutable { counter.increment(ticks); };
}
}  // namespace srf::internal::segment

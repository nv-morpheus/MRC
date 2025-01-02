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

#include "mrc/options/options.hpp"

#include "mrc/options/engine_groups.hpp"
#include "mrc/options/fiber_pool.hpp"
#include "mrc/options/placement.hpp"
#include "mrc/options/resources.hpp"
#include "mrc/options/services.hpp"
#include "mrc/options/topology.hpp"

#include <glog/logging.h>

#include <utility>  // for move

namespace mrc {

Options::Options() :
  m_engine_groups(std::make_unique<EngineGroups>()),
  m_fiber_pool(std::make_unique<FiberPoolOptions>()),
  m_placement(std::make_unique<PlacementOptions>()),
  m_resources(std::make_unique<ResourceOptions>()),
  m_services(std::make_unique<ServiceOptions>()),
  m_topology(std::make_unique<TopologyOptions>())
{}

Options::Options(const Options& other) :
  m_engine_groups(std::make_unique<EngineGroups>(*other.m_engine_groups)),
  m_fiber_pool(std::make_unique<FiberPoolOptions>(*other.m_fiber_pool)),
  m_placement(std::make_unique<PlacementOptions>(*other.m_placement)),
  m_resources(std::make_unique<ResourceOptions>(*other.m_resources)),
  m_services(std::make_unique<ServiceOptions>(*other.m_services)),
  m_topology(std::make_unique<TopologyOptions>(*other.m_topology)),
  m_architect_url(other.m_architect_url),
  m_enable_server(other.m_enable_server),
  m_server_port(other.m_server_port),
  m_config_request(other.m_config_request)
{}

Options& Options::operator=(const Options& other)
{
    // protect against invalid self-assignment
    if (this != &other)
    {
        // Child objects
        *m_engine_groups = *other.m_engine_groups;
        *m_fiber_pool    = *other.m_fiber_pool;
        *m_placement     = *other.m_placement;
        *m_resources     = *other.m_resources;
        *m_services      = *other.m_services;
        *m_topology      = *other.m_topology;

        // Values
        m_architect_url  = other.m_architect_url;
        m_enable_server  = other.m_enable_server;
        m_server_port    = other.m_server_port;
        m_config_request = other.m_config_request;
    }

    return *this;
}

TopologyOptions& Options::topology()
{
    CHECK(m_topology);
    return *m_topology;
}
const TopologyOptions& Options::topology() const
{
    CHECK(m_topology);
    return *m_topology;
}

FiberPoolOptions& Options::fiber_pool()
{
    CHECK(m_fiber_pool);
    return *m_fiber_pool;
}
const FiberPoolOptions& Options::fiber_pool() const
{
    CHECK(m_fiber_pool);
    return *m_fiber_pool;
}

PlacementOptions& Options::placement()
{
    CHECK(m_placement);
    return *m_placement;
}
const PlacementOptions& Options::placement() const
{
    CHECK(m_placement);
    return *m_placement;
}

ResourceOptions& Options::resources()
{
    CHECK(m_resources);
    return *m_resources;
}
const ResourceOptions& Options::resources() const
{
    CHECK(m_resources);
    return *m_resources;
}

const std::string& Options::architect_url() const
{
    return m_architect_url;
}

void Options::architect_url(std::string url)
{
    m_architect_url = std::move(url);
}

void Options::enable_server(bool default_false)
{
    m_enable_server = default_false;
}

bool Options::enable_server() const
{
    return m_enable_server;
}

const std::string& Options::config_request() const
{
    return m_config_request;
}

void Options::config_request(std::string config_request)
{
    m_config_request = config_request;
}

ServiceOptions& Options::services()
{
    CHECK(m_services);
    return *m_services;
}

const ServiceOptions& Options::services() const
{
    CHECK(m_services);
    return *m_services;
}

EngineGroups& Options::engine_factories()
{
    CHECK(m_engine_groups);
    return *m_engine_groups;
}

const EngineGroups& Options::engine_factories() const
{
    CHECK(m_engine_groups);
    return *m_engine_groups;
}

std::uint16_t Options::server_port() const
{
    return m_server_port;
}
void Options::server_port(std::uint16_t port)
{
    m_server_port = port;
}
}  // namespace mrc

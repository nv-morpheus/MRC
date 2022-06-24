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

#pragma once

#include "srf/options/engine_groups.hpp"
#include "srf/options/fiber_pool.hpp"
#include "srf/options/placement.hpp"
#include "srf/options/resources.hpp"
#include "srf/options/services.hpp"
#include "srf/options/topology.hpp"

#include <memory>
#include <string>

namespace srf {

class Options
{
  public:
    Options();

    EngineGroups& engine_factories();
    FiberPoolOptions& fiber_pool();
    PlacementOptions& placement();
    ResourceOptions& resources();
    ServiceOptions& services();
    TopologyOptions& topology();

    void architect_url(std::string url);
    void enable_server(bool default_false = false);
    void config_request(std::string config);

    [[nodiscard]] const EngineGroups& engine_factories() const;
    [[nodiscard]] const FiberPoolOptions& fiber_pool() const;
    [[nodiscard]] const PlacementOptions& placement() const;
    [[nodiscard]] const ResourceOptions& resources() const;
    [[nodiscard]] const ServiceOptions& services() const;
    [[nodiscard]] const TopologyOptions& topology() const;

    [[nodiscard]] const std::string& architect_url() const;
    [[nodiscard]] const std::string& config_request() const;
    [[nodiscard]] bool enable_server() const;

  private:
    std::unique_ptr<EngineGroups> m_engine_groups;
    std::unique_ptr<FiberPoolOptions> m_fiber_pool;
    std::unique_ptr<PlacementOptions> m_placement;
    std::unique_ptr<ResourceOptions> m_resources;
    std::unique_ptr<ServiceOptions> m_services;
    std::unique_ptr<TopologyOptions> m_topology;

    std::string m_architect_url;
    bool m_enable_server{false};
    std::string m_config_request{"*:1:*"};
};

}  // namespace srf

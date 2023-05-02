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

#include "internal/runtime/runtime.hpp"

#include "internal/control_plane/client.hpp"
#include "internal/resources/manager.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/runtime/partition_manager.hpp"
#include "internal/runtime/partition_runtime.hpp"
#include "internal/runtime/pipelines_manager.hpp"
#include "internal/system/partitions.hpp"
#include "internal/system/system_provider.hpp"

#include "mrc/protos/architect.pb.h"
#include "mrc/types.hpp"

#include <glog/logging.h>

#include <memory>
#include <optional>
#include <utility>

namespace mrc::internal::runtime {

// Runtime::Runtime(std::unique_ptr<resources::Manager> resources) : m_resources(std::move(resources))
// {
//     CHECK(m_resources);
//     for (int i = 0; i < m_resources->partition_count(); i++)
//     {
//         m_partitions.push_back(std::make_unique<Partition>(m_resources->partition(i)));
//     }
// }

Runtime::Runtime(const system::SystemProvider& system) : system::SystemProvider(system) {}

Runtime::~Runtime()
{
    // the problem is that m_partitions goes away, then m_resources is destroyed
    // when not all Publishers/Subscribers which were created with a ref to a Partition
    // might not yet be finished
    m_resources->shutdown().get();

    Service::call_in_destructor();
}

resources::Manager& Runtime::resources() const
{
    CHECK(m_resources);
    return *m_resources;
}
std::size_t Runtime::partition_count() const
{
    return m_partitions.size();
}

std::size_t Runtime::gpu_count() const
{
    return resources().device_count();
}

PartitionRuntime& Runtime::partition(std::size_t partition_id)
{
    DCHECK_LT(partition_id, m_resources->partition_count());
    DCHECK(m_partitions.at(partition_id));
    return *m_partitions.at(partition_id);
}

control_plane::Client& Runtime::control_plane() const
{
    return *m_control_plane_client;
}

void Runtime::register_pipelines_defs(std::map<int, std::shared_ptr<pipeline::Pipeline>> pipeline_defs)
{
    // Save the pipeline definitions
}

void Runtime::do_service_start()
{
    // First, optionally create the control plane server
    if (system().options().architect_url().empty())
    {
        if (system().options().enable_server())
        {
            m_control_plane_server = std::make_unique<control_plane::Server>();
            m_control_plane_server->service_start();
            m_control_plane_server->service_await_live();
        }
        else
        {
            LOG(WARNING) << "No Architect URL has been specified but enable_server = false. Ensure you know what you "
                            "are doing";
        }
    }

    // Create the system resources first
    auto sys_resources = std::make_unique<system::SystemResources>(*this);

    // Now create the control plane client
    auto runnable          = runnable::RunnableResources(*sys_resources, 0);
    auto part_base         = resources::PartitionResourceBase(runnable, 0);
    m_control_plane_client = std::make_unique<control_plane::Client>(part_base);
    m_control_plane_client->service_start();
    m_control_plane_client->service_await_live();

    // Create/Initialize the runtime resources object (Could go before the control plane client)
    m_resources = std::make_unique<resources::Manager>(std::move(sys_resources));
    m_resources->initialize();

    // For each partition, create and start a partition manager
    for (size_t i = 0; i < m_resources->partition_count(); i++)
    {
        m_partition_managers.emplace_back(std::make_unique<PartitionManager>(m_resources->partition(i),
                                                                             *m_control_plane_client,
                                                                             *m_pipelines_manager));

        m_partition_managers.back()->service_start();
    }

    // Now ensure they are all alive
    for (auto& part_manager : m_partition_managers)
    {
        part_manager->service_await_live();
    }

    // Finally, create the pipelines manager
    m_pipelines_manager = std::make_unique<PipelinesManager>(*m_control_plane_client);
}

void Runtime::do_service_stop()
{
    for (auto& part : m_partition_managers)
    {
        part->service_stop();
    }

    if (m_control_plane_client)
    {
        m_control_plane_client->service_stop();
    }

    if (m_control_plane_server)
    {
        m_control_plane_server->service_stop();
    }
}

void Runtime::do_service_kill()
{
    for (auto& part : m_partition_managers)
    {
        part->service_kill();
    }

    if (m_control_plane_client)
    {
        m_control_plane_client->service_kill();
    }

    if (m_control_plane_server)
    {
        m_control_plane_server->service_kill();
    }
}

void Runtime::do_service_await_live()
{
    for (auto& part : m_partition_managers)
    {
        part->service_await_live();
    }

    if (m_control_plane_client)
    {
        m_control_plane_client->service_await_live();
    }

    if (m_control_plane_server)
    {
        m_control_plane_server->service_await_live();
    }
}

void Runtime::do_service_await_join()
{
    for (auto& part : m_partition_managers)
    {
        part->service_await_join();
    }

    if (m_control_plane_client)
    {
        m_control_plane_client->service_await_join();
    }

    if (m_control_plane_server)
    {
        m_control_plane_server->service_await_join();
    }
}
}  // namespace mrc::internal::runtime

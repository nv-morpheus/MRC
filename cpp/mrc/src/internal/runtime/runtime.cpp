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

#include "internal/async_service.hpp"
#include "internal/control_plane/client.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/runtime/partition_runtime.hpp"
#include "internal/runtime/pipelines_manager.hpp"
#include "internal/runtime/segments_manager.hpp"
#include "internal/system/partitions.hpp"
#include "internal/system/system_provider.hpp"

#include "mrc/protos/architect.pb.h"
#include "mrc/types.hpp"

#include <glog/logging.h>

#include <memory>
#include <optional>
#include <utility>

namespace mrc::internal::runtime {

Runtime::Runtime(std::unique_ptr<resources::SystemResources> resources) :
  AsyncService("Runtime"),
  system::SystemProvider(*resources),
  m_sys_resources(std::move(resources))
{
    CHECK(m_sys_resources) << "resources cannot be null";
    // for (int i = 0; i < m_sys_resources->partition_count(); i++)
    // {
    //     m_partitions.push_back(std::make_unique<Partition>(m_sys_resources->partition(i)));
    // }

    // // Now create the threading resources and system wide runnable so it is available to AsyncService
    // m_sys_threading_resources = std::make_unique<system::ThreadingResources>(*this);

    // m_sys_runnable_resources = std::make_unique<runnable::RunnableResources>(*m_sys_threading_resources, 0);
}

// Call the other constructor with a new ThreadingResources
Runtime::Runtime(const system::SystemProvider& system) : Runtime(std::make_unique<resources::SystemResources>(system))
{}

Runtime::~Runtime()
{
    // the problem is that m_partitions goes away, then m_sys_resources is destroyed
    // when not all Publishers/Subscribers which were created with a ref to a Partition
    // might not yet be finished
    // m_sys_resources->shutdown().get();

    AsyncService::call_in_destructor();
}

resources::SystemResources& Runtime::resources() const
{
    CHECK(m_sys_resources);
    return *m_sys_resources;
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
    DCHECK_LT(partition_id, m_sys_resources->partition_count());
    DCHECK(m_partitions.at(partition_id));
    return *m_partitions.at(partition_id);
}

control_plane::Client& Runtime::control_plane() const
{
    return *m_control_plane_client;
}

PipelinesManager& Runtime::pipelines_manager() const
{
    return *m_pipelines_manager;
}

metrics::Registry& Runtime::metrics_registry() const
{
    return *m_metrics_registry;
}

runnable::RunnableResources& Runtime::runnable()
{
    return m_sys_resources->runnable();
}

void Runtime::do_service_start(std::stop_token stop_token)
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

    // // Create the system resources first
    // auto sys_resources = std::make_unique<system::ThreadingResources>(*this);

    // // Now create the control plane client
    // auto runnable          = runnable::RunnableResources(*sys_resources, 0);
    // auto part_base         = resources::PartitionResourceBase(runnable, 0);

    m_control_plane_client = std::make_unique<control_plane::Client>(*m_sys_resources);
    m_control_plane_client->service_start();
    m_control_plane_client->service_await_live();

    // // Create/Initialize the runtime resources object (Could go before the control plane client)
    // m_sys_resources = std::make_unique<resources::SystemResources>(std::move(sys_resources));
    // m_sys_resources->initialize();

    // Before creating the partitions, create the pipelines manager
    m_pipelines_manager = std::make_unique<PipelinesManager>(*this);

    this->child_service_start(*m_pipelines_manager);

    // For each partition, create and start a partition manager
    for (size_t i = 0; i < m_sys_resources->partition_count(); i++)
    {
        auto& part_runtime = m_partitions.emplace_back(std::make_unique<PartitionRuntime>(*this, i));

        this->child_service_start(*part_runtime);
    }

    // // Now ensure they are all alive
    // for (auto& part_manager : m_partition_managers)
    // {
    //     part_manager->service_await_live();
    // }

    // Indicate we have started (forces children to be ready before returning)
    this->mark_started();

    // Exit here. This will rely on the children to determine when we are completed
}

// void Runtime::do_service_start()
// {
//     // First, optionally create the control plane server
//     if (system().options().architect_url().empty())
//     {
//         if (system().options().enable_server())
//         {
//             m_control_plane_server = std::make_unique<control_plane::Server>();
//             m_control_plane_server->service_start();
//             m_control_plane_server->service_await_live();
//         }
//         else
//         {
//             LOG(WARNING) << "No Architect URL has been specified but enable_server = false. Ensure you know what you
//             "
//                             "are doing";
//         }
//     }

//     // Create the system resources first
//     auto sys_resources = std::make_unique<system::ThreadingResources>(*this);

//     // Now create the control plane client
//     auto runnable          = runnable::RunnableResources(*sys_resources, 0);
//     auto part_base         = resources::PartitionResourceBase(runnable, 0);
//     m_control_plane_client = std::make_unique<control_plane::Client>(part_base);
//     m_control_plane_client->service_start();
//     m_control_plane_client->service_await_live();

//     // Create/Initialize the runtime resources object (Could go before the control plane client)
//     m_sys_resources = std::make_unique<resources::SystemResources>(std::move(sys_resources));
//     m_sys_resources->initialize();

//     // For each partition, create and start a partition manager
//     for (size_t i = 0; i < m_sys_resources->partition_count(); i++)
//     {
//         m_partitions.emplace_back(std::make_unique<PartitionRuntime>(*this, i));

//         m_partition_managers.emplace_back(std::make_unique<SegmentsManager>(this->partition(i)));

//         m_partition_managers.back()->service_start();
//     }

//     // Now ensure they are all alive
//     for (auto& part_manager : m_partition_managers)
//     {
//         part_manager->service_await_live();
//     }

//     // Finally, create the pipelines manager
//     m_pipelines_manager = std::make_unique<PipelinesManager>(*m_control_plane_client);
// }

// void Runtime::do_service_stop()
// {
//     for (auto& part : m_partition_managers)
//     {
//         part->service_stop();
//     }

//     if (m_control_plane_client)
//     {
//         m_control_plane_client->service_stop();
//     }

//     if (m_control_plane_server)
//     {
//         m_control_plane_server->service_stop();
//     }
// }

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

// void Runtime::do_service_await_live()
// {
//     for (auto& part : m_partition_managers)
//     {
//         part->service_await_live();
//     }

//     if (m_control_plane_client)
//     {
//         m_control_plane_client->service_await_live();
//     }

//     if (m_control_plane_server)
//     {
//         m_control_plane_server->service_await_live();
//     }
// }

// void Runtime::do_service_await_join()
// {
//     for (auto& part : m_partition_managers)
//     {
//         part->service_await_join();
//     }

//     if (m_control_plane_client)
//     {
//         m_control_plane_client->service_await_join();
//     }

//     if (m_control_plane_server)
//     {
//         m_control_plane_server->service_await_join();
//     }
// }

}  // namespace mrc::internal::runtime

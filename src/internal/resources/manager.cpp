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

#include "internal/resources/manager.hpp"

#include "internal/data_plane/resources.hpp"
#include "internal/resources/forward.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/system/partition.hpp"
#include "internal/system/partitions.hpp"
#include "internal/system/system.hpp"
#include "internal/ucx/registation_callback_builder.hpp"

#include "srf/exceptions/runtime_error.hpp"
#include "srf/options/options.hpp"
#include "srf/options/placement.hpp"

#include <ext/alloc_traits.h>
#include <glog/logging.h>

#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <thread>
#include <utility>

namespace srf::internal::resources {

thread_local Manager* Manager::m_thread_resources{nullptr};
thread_local PartitionResources* Manager::m_thread_partition{nullptr};

Manager::Manager(const system::SystemProvider& system) : Manager(std::make_unique<system::Resources>(system)) {}

Manager::Manager(std::unique_ptr<system::Resources> resources) :
  SystemProvider(*resources),
  m_system(std::move(resources))
{
    const auto& partitions      = system().partitions().flattened();
    const auto& host_partitions = system().partitions().host_partitions();
    const bool network_enabled  = !system().options().architect_url().empty();

    // construct the runnable resources on each host_partition - launch control and main
    for (std::size_t i = 0; i < host_partitions.size(); ++i)
    {
        VLOG(1) << "building runnable/launch_control resources on host_partition: " << i;
        m_runnable.emplace_back(*m_system, i);
    }

    std::vector<PartitionResourceBase> base_partition_resources;
    for (int i = 0; i < partitions.size(); i++)
    {
        auto host_partition_id = partitions.at(i).host_partition_id();
        base_partition_resources.emplace_back(m_runnable.at(host_partition_id), i);
    }

    // construct ucx resources on each flattened partition
    // this provides a ucx context, ucx worker and registration cache per partition
    for (auto& base : base_partition_resources)
    {
        if (network_enabled)
        {
            VLOG(1) << "building ucx resources for partition " << base.partition_id();
            auto network_task_queue_cpuset =
                base.partition().host().engine_factory_cpu_sets().fiber_cpu_sets.at("srf_network");
            auto& network_fiber_queue = m_system->get_task_queue(network_task_queue_cpuset.first());
            std::optional<ucx::Resources> ucx;
            ucx.emplace(base, network_fiber_queue);
            m_ucx.push_back(std::move(ucx));
        }
        else
        {
            m_ucx.emplace_back(std::nullopt);
        }
    }

    // construct the host memory resources for each host_partition
    for (std::size_t i = 0; i < host_partitions.size(); ++i)
    {
        ucx::RegistrationCallbackBuilder builder;
        for (auto& ucx : m_ucx)
        {
            if (ucx)
            {
                if (ucx->partition().host_partition_id() == i)
                {
                    ucx->add_registration_cache_to_builder(builder);
                }
            }
        }
        VLOG(1) << "building host resources for host_partition: " << i;
        m_host.emplace_back(m_runnable.at(i), std::move(builder));
    }

    // devices
    for (auto& base : base_partition_resources)
    {
        VLOG(1) << "building device resources for partition: " << base.partition_id();
        if (base.partition().has_device())
        {
            DCHECK_LT(base.partition_id(), device_count());
            std::optional<memory::DeviceResources> device;
            device.emplace(base, m_ucx.at(base.partition_id()));
            m_device.emplace_back(std::move(device));
        }
        else
        {
            m_device.emplace_back(std::nullopt);
        }
    }

    // network resources
    for (auto& base : base_partition_resources)
    {
        if (network_enabled)
        {
            VLOG(1) << "building network resources for partition: " << base.partition_id();
            CHECK(m_ucx.at(base.partition_id()));
            std::optional<network::Resources> network;
            network.emplace(base, *m_ucx.at(base.partition_id()), m_host.at(base.partition().host_partition_id()));
            m_network.push_back(std::move(network));
        }
        else
        {
            m_network.emplace_back(std::nullopt);
        }
    }

    // partition resources
    for (std::size_t i = 0; i < partition_count(); ++i)
    {
        VLOG(1) << "building partition_resources for partition: " << i;
        auto host_partition_id = partitions.at(i).host_partition_id();
        m_partitions.emplace_back(
            m_runnable.at(host_partition_id), i, m_host.at(host_partition_id), m_device.at(i), m_network.at(i));
    }

    // set thread local access to resources on all fiber task queues and any future thread created by the runtime
    for (auto& partition : m_partitions)
    {
        m_system->register_thread_local_initializer(partition.partition().host().cpu_set(), [this, &partition] {
            m_thread_resources = this;
            if (system().partitions().device_to_host_strategy() == PlacementResources::Dedicated)
            {
                m_thread_partition = &partition;
            }
        });
    }
}

std::size_t Manager::partition_count() const
{
    return system().partitions().flattened().size();
};

std::size_t Manager::device_count() const
{
    return system().partitions().device_partitions().size();
};

PartitionResources& Manager::partition(std::size_t partition_id)
{
    CHECK_LT(partition_id, m_partitions.size());
    return m_partitions.at(partition_id);
}

Manager& Manager::get_resources()
{
    if (m_thread_resources == nullptr)  // todo(cpp20) [[unlikely]]
    {
        LOG(ERROR) << "thread with id " << std::this_thread::get_id()
                   << " attempting to access the SRF runtime, but is not a SRF runtime thread";
        throw exceptions::SrfRuntimeError("can not access runtime resources from outside the runtime");
    }

    return *m_thread_resources;
}

PartitionResources& Manager::get_partition()
{
    {
        if (m_thread_partition == nullptr)  // todo(cpp20) [[unlikely]]
        {
            auto& resources = Manager::get_resources();

            if (resources.system().partitions().device_to_host_strategy() == PlacementResources::Shared)
            {
                LOG(ERROR) << "runtime partition query is disabed when PlacementResources::Shared is in use";
                throw exceptions::SrfRuntimeError("unable to access runtime parition info");
            }

            LOG(ERROR) << "thread with id " << std::this_thread::get_id()
                       << " attempting to access the SRF runtime, but is not a SRF runtime thread";
            throw exceptions::SrfRuntimeError("can not access runtime resources from outside the runtime");
        }

        return *m_thread_partition;
    }
}
}  // namespace srf::internal::resources

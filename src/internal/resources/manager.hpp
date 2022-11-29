/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/memory/device_resources.hpp"
#include "internal/memory/host_resources.hpp"
#include "internal/network/resources.hpp"
#include "internal/resources/forward.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/system/resources.hpp"
#include "internal/system/system_provider.hpp"
#include "internal/ucx/resources.hpp"

#include "mrc/core/task_queue.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace mrc::internal::runtime {
class Runtime;
}

namespace mrc::internal::resources {

class Manager final : public system::SystemProvider
{
  public:
    Manager(const system::SystemProvider& system);
    Manager(std::unique_ptr<system::Resources> resources);
    ~Manager() override;

    static Manager& get_resources();
    static PartitionResources& get_partition();

    std::size_t device_count() const;
    std::size_t partition_count() const;

    PartitionResources& partition(std::size_t partition_id);

  private:
    Future<void> shutdown()
    {
        return m_runnable.at(0).main().enqueue([this] {
            std::vector<Future<void>> futures;
            futures.reserve(m_network.size());
            for (auto& net : m_network)
            {
                if (net)
                {
                    futures.emplace_back(net->shutdown());
                }
            }
            for (auto& f : futures)
            {
                f.get();
            }
        });
    }

    const std::unique_ptr<system::Resources> m_system;
    std::vector<runnable::Resources> m_runnable;                   // one per host partition
    std::vector<std::optional<ucx::Resources>> m_ucx;              // one per flattened partition if network is enabled
    std::shared_ptr<control_plane::Resources> m_control_plane;     // one per instance of resources::Manager
    std::vector<memory::HostResources> m_host;                     // one per host partition
    std::vector<std::optional<memory::DeviceResources>> m_device;  // one per flattened partition upto device_count
    std::vector<PartitionResources> m_partitions;                  // one per flattened partition

    // this is the final variable in the list
    // so it can be the first variable destroyed
    // this is the owner of the control_plane::Client::Instance
    // which must be destroyed before all other
    std::vector<std::optional<network::Resources>> m_network;  // one per flattened partition

    static thread_local PartitionResources* m_thread_partition;
    static thread_local Manager* m_thread_resources;

    friend runtime::Runtime;
};

}  // namespace mrc::internal::resources

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

#pragma once

#include "internal/memory/host_resources.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/runnable/runnable_resources.hpp"
#include "internal/system/system_provider.hpp"

#include "mrc/types.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>
// IWYU pragma: no_include "internal/memory/device_resources.hpp"
// IWYU pragma: no_include "internal/network/network_resources.hpp"
// IWYU pragma: no_include "internal/ucx/ucx_resources.hpp"

namespace mrc::network {
class NetworkResources;  // IWYU pragma: keep
}  // namespace mrc::network
namespace mrc::control_plane {
class ControlPlaneResources;
}  // namespace mrc::control_plane
namespace mrc::memory {
class DeviceResources;  // IWYU pragma: keep
}  // namespace mrc::memory
namespace mrc::system {
class ThreadingResources;
}  // namespace mrc::system
namespace mrc::ucx {
class UcxResources;  // IWYU pragma: keep
}  // namespace mrc::ucx
namespace mrc::runtime {
class Runtime;
}  // namespace mrc::runtime

namespace mrc::resources {

class Manager final : public system::SystemProvider
{
  public:
    Manager(const system::SystemProvider& system);
    // Manager(std::unique_ptr<system::ThreadingResources> resources);
    ~Manager() override;

    std::size_t runtime_id() const;

    static Manager& get_resources();
    static PartitionResources& get_partition();

    std::size_t device_count() const;
    std::size_t partition_count() const;

    PartitionResources& partition(std::size_t partition_id);

  private:
    Future<void> shutdown();

    const size_t m_runtime_id;  // unique id for this runtime

    const std::unique_ptr<system::ThreadingResources> m_threading;
    std::vector<runnable::RunnableResources> m_runnable;  // one per host partition
    std::vector<std::optional<ucx::UcxResources>> m_ucx;  // one per flattened partition if network is enabled
    std::shared_ptr<control_plane::ControlPlaneResources> m_control_plane;  // one per instance of resources::Manager
    std::vector<memory::HostResources> m_host;                              // one per host partition
    std::vector<std::optional<memory::DeviceResources>> m_device;  // one per flattened partition upto device_count
    std::vector<PartitionResources> m_partitions;                  // one per flattened partition

    // this is the final variable in the list
    // so it can be the first variable destroyed
    // this is the owner of the control_plane::Client::Instance
    // which must be destroyed before all other
    std::vector<std::optional<network::NetworkResources>> m_network;  // one per flattened partition

    static std::atomic_size_t s_id_counter;
    static thread_local PartitionResources* m_thread_partition;
    static thread_local Manager* m_thread_resources;

    friend runtime::Runtime;
};

}  // namespace mrc::resources

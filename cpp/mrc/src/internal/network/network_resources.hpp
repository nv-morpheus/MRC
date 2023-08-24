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

#pragma once

#include "internal/resources/partition_resources_base.hpp"

#include "mrc/types.hpp"
#include "mrc/utils/macros.hpp"

#include <memory>

namespace mrc::control_plane {
class Client;
}  // namespace mrc::control_plane
namespace mrc::control_plane::client {
class Instance;
}  // namespace mrc::control_plane::client
namespace mrc::data_plane {
class DataPlaneResources;
}  // namespace mrc::data_plane
namespace mrc::memory {
class HostResources;
}  // namespace mrc::memory
namespace mrc::resources {
class SystemResources;
}  // namespace mrc::resources
namespace mrc::ucx {
class UcxResources;
}  // namespace mrc::ucx

namespace mrc::network {

class NetworkResources final : private resources::PartitionResourceBase
{
  public:
    NetworkResources(resources::PartitionResourceBase& base,
                     ucx::UcxResources& ucx,
                     memory::HostResources& host,
                     std::unique_ptr<control_plane::client::Instance> control_plane);
    ~NetworkResources() final;

    DELETE_COPYABILITY(NetworkResources);

    // todo(clang-format-15)
    // clang-format off
    NetworkResources(NetworkResources&&) noexcept            = default;
    NetworkResources& operator=(NetworkResources&&) noexcept = delete;
    // clang-format on

    const InstanceID& instance_id() const;

    ucx::UcxResources& ucx();
    control_plane::client::Instance& control_plane();
    data_plane::DataPlaneResources& data_plane();

  private:
    Future<void> shutdown();

    InstanceID m_instance_id;
    ucx::UcxResources& m_ucx;
    control_plane::Client& m_control_plane_client;
    std::unique_ptr<data_plane::DataPlaneResources> m_data_plane;

    // this must be the first variable destroyed
    std::unique_ptr<control_plane::client::Instance> m_control_plane;

    friend resources::SystemResources;
};

}  // namespace mrc::network

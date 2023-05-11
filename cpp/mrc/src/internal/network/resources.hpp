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

namespace mrc::internal::control_plane {
class Client;
}  // namespace mrc::internal::control_plane
namespace mrc::internal::control_plane::client {
class Instance;
}  // namespace mrc::internal::control_plane::client
namespace mrc::internal::data_plane {
class DataPlaneResources;
}  // namespace mrc::internal::data_plane
namespace mrc::internal::memory {
class HostResources;
}  // namespace mrc::internal::memory
namespace mrc::internal::resources {
class Manager;
}  // namespace mrc::internal::resources
namespace mrc::internal::ucx {
class Resources;
}  // namespace mrc::internal::ucx

namespace mrc::internal::network {

class Resources final : private resources::PartitionResourceBase
{
  public:
    Resources(resources::PartitionResourceBase& base,
              ucx::Resources& ucx,
              memory::HostResources& host,
              std::unique_ptr<control_plane::client::Instance> control_plane);
    ~Resources() final;

    DELETE_COPYABILITY(Resources);

    // todo(clang-format-15)
    // clang-format off
    Resources(Resources&&) noexcept            = default;
    Resources& operator=(Resources&&) noexcept = delete;
    // clang-format on

    const InstanceID& instance_id() const;

    ucx::Resources& ucx();
    control_plane::client::Instance& control_plane();
    data_plane::DataPlaneResources& data_plane();

  private:
    Future<void> shutdown();

    InstanceID m_instance_id;
    ucx::Resources& m_ucx;
    control_plane::Client& m_control_plane_client;
    std::unique_ptr<data_plane::DataPlaneResources> m_data_plane;

    // this must be the first variable destroyed
    std::unique_ptr<control_plane::client::Instance> m_control_plane;

    friend resources::Manager;
};

}  // namespace mrc::internal::network

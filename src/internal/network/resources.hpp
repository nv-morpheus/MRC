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

#include "internal/control_plane/client/state_manager.hpp"
#include "internal/remote_descriptor/remote_descriptor.hpp"
#include "internal/resources/forward.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/ucx/registration_cache.hpp"
#include "internal/ucx/resources.hpp"

#include "srf/utils/macros.hpp"

namespace srf::internal::network {

class Resources final : private resources::PartitionResourceBase
{
  public:
    Resources(resources::PartitionResourceBase& base,
              ucx::Resources& ucx,
              memory::HostResources& host,
              std::unique_ptr<control_plane::client::Instance> control_plane);
    ~Resources() final;

    DELETE_COPYABILITY(Resources);

    Resources(Resources&&) noexcept            = default;
    Resources& operator=(Resources&&) noexcept = delete;

    const InstanceID& instance_id() const;

    ucx::Resources& ucx();
    control_plane::client::Instance& control_plane();
    data_plane::Resources& data_plane();

  private:
    Future<void> shutdown();

    InstanceID m_instance_id;
    ucx::Resources& m_ucx;
    control_plane::Client& m_control_plane_client;
    std::unique_ptr<data_plane::Resources> m_data_plane;

    // this must be the first variable destroyed
    std::unique_ptr<control_plane::client::Instance> m_control_plane;

    friend resources::Manager;
};

}  // namespace srf::internal::network

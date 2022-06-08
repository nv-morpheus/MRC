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

#include "internal/resources/device_resources.hpp"
#include "internal/resources/host_resources.hpp"
#include "internal/system/forward.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace srf::internal::resources {

using system::System;

class SystemResources
{
  public:
    static std::shared_ptr<SystemResources> create(std::shared_ptr<System> system);

    SystemResources(std::shared_ptr<System> system);
    virtual ~SystemResources() = default;

    const std::vector<std::shared_ptr<HostResources>>& host_resources() const;
    const std::vector<std::shared_ptr<DeviceResources>>& device_resources() const;

  protected:
    System& system() const;

  private:
    std::shared_ptr<System> m_system;
    std::vector<std::shared_ptr<HostResources>> m_host_resources;
    std::vector<std::shared_ptr<DeviceResources>> m_device_resources;
};

inline std::shared_ptr<SystemResources> make_system_resources(std::shared_ptr<system::System> system)
{
    return SystemResources::create(std::move(system));
}

}  // namespace srf::internal::resources

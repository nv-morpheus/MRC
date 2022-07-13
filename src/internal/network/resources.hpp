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

#include "internal/memory/device_resources.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/ucx/registration_cache.hpp"

#include <cstddef>

namespace srf::internal {
namespace ucx {
class Resources;
}  // namespace ucx
namespace memory {
class HostResources;
}  // namespace memory
namespace data_plane {
class Client;
class Server;
}  // namespace data_plane
}  // namespace srf::internal

namespace srf::internal::network {

class Resources final : private resources::PartitionResourceBase
{
  public:
    Resources(runnable::Resources& _runnable_resources,
              std::size_t _partition_id,
              ucx::Resources& ucx,
              memory::HostResources& host);
    ~Resources() final;

    const ucx::RegistrationCache& registration_cache() const;

  private:
    ucx::Resources& m_ucx;
    memory::HostResources& m_host;
    std::shared_ptr<data_plane::Server> m_server;
    std::shared_ptr<data_plane::Client> m_client;
};

}  // namespace srf::internal::network

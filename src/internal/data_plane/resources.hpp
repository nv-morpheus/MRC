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

#include "internal/data_plane/client.hpp"
#include "internal/data_plane/server.hpp"
#include "internal/resources/forward.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/service.hpp"

#include "srf/protos/codable.pb.h"

#include <memory>

namespace srf::internal::data_plane {

/**
 * @brief ArchitectResources hold and is responsible for constructing any object that depending the UCX data plane
 *
 */
class Resources final : private Service, private resources::PartitionResourceBase
{
  public:
    Resources(resources::PartitionResourceBase& base, ucx::Resources& ucx, memory::HostResources& host);
    ~Resources() final;

    Client& client();

    std::string ucx_address() const;
    const ucx::RegistrationCache& registration_cache() const;

  private:
    void do_service_start() final;
    void do_service_await_live() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_join() final;

    ucx::Resources& m_ucx;
    memory::HostResources& m_host;

    Server m_server;
    Client m_client;

    friend network::Resources;
};

}  // namespace srf::internal::data_plane

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

#include "internal/service.hpp"
#include "internal/ucx/context.hpp"
#include "srf/runnable/launch_control.hpp"

#include <memory>

namespace srf::internal::data_plane {

/**
 * @brief ArchitectResources hold and is responsible for constructing any object that depending the UCX data plane
 *
 */
class Instance final : public Service
{
  public:
    Instance(std::shared_ptr<resources::PartitionResources> resources);
    ~Instance() final;

    Client& client() const;
    Server& server() const;

  private:
    Service& client_service();
    Service& server_service();

    void do_service_start() final;
    void do_service_await_live() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_join() final;

    std::shared_ptr<resources::PartitionResources> m_resources;
    std::shared_ptr<ucx::Context> m_context;
    std::unique_ptr<Client> m_client;
    std::unique_ptr<Server> m_server;
};

}  // namespace srf::internal::data_plane

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

#include "internal/control_plane/control_plane_resources.hpp"

#include "internal/control_plane/client.hpp"
#include "internal/control_plane/server.hpp"
#include "internal/system/system.hpp"

#include "mrc/options/options.hpp"

#include <glog/logging.h>

#include <memory>

namespace mrc::control_plane {

ControlPlaneResources::ControlPlaneResources(resources::PartitionResourceBase& base) :
  resources::PartitionResourceBase(base),
  m_client(std::make_unique<control_plane::Client>(base))
{
    if (system().options().architect_url().empty())
    {
        if (system().options().enable_server())
        {
            m_server = std::make_unique<Server>();
            m_server->service_start();
            m_server->service_await_live();
        }
        else
        {
            LOG(WARNING) << "No Architect URL has been specified but enable_server = false. Ensure you know what you "
                            "are doing";
        }
    }

    CHECK(m_client);
    m_client->service_start();
    m_client->service_await_live();
}

ControlPlaneResources::~ControlPlaneResources()
{
    if (m_client)
    {
        m_client->service_stop();
        m_client->service_await_join();
    }

    if (m_server)
    {
        m_server->service_stop();
        m_server->service_await_join();
    }
}

}  // namespace mrc::control_plane

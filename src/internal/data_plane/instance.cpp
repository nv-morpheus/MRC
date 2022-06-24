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

#include "internal/data_plane/instance.hpp"

#include "internal/data_plane/client.hpp"
#include "internal/data_plane/server.hpp"
#include "internal/ucx/context.hpp"

#include <srf/runnable/launch_control.hpp>

#include <glog/logging.h>

#include <memory>

namespace srf::internal::data_plane {

Instance::Instance(std::unique_ptr<resources::PartitionResources> resources) : m_resources(std::move(resources)) {}
// m_context(std::make_shared<ucx::Context>()),
// m_client(std::make_shared<Client>(m_context)),
// m_server(std::make_shared<Server>(m_context))
// {}

Instance::~Instance()
{
    call_in_destructor();
}

Client& Instance::comms_manager() const
{
    CHECK(m_client);
    return *m_client;
}

Server& Instance::events_manager() const
{
    CHECK(m_server);
    return *m_server;
}

Service& Instance::iclient()
{
    CHECK(m_client);
    return *m_client;
}

Service& Instance::iserver()
{
    CHECK(m_server);
    return *m_server;
}

void Instance::do_service_start()
{
    // iclient().service_start(launch_control);
    // iserver().service_start(launch_control);
}

void Instance::do_service_await_live()
{
    iclient().service_await_live();
    iserver().service_await_live();
}

void Instance::do_service_stop()
{
    iclient().service_stop();
    iserver().service_stop();
}

void Instance::do_service_kill()
{
    iclient().service_kill();
    iserver().service_kill();
}

void Instance::do_service_await_join()
{
    iclient().service_await_join();
    iserver().service_await_join();
}

}  // namespace srf::internal::data_plane

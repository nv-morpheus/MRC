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

#include <srf/cuda/common.hpp>
#include <srf/runnable/launch_control.hpp>
#include "internal/ucx/context.hpp"

#include <glog/logging.h>

#include <memory>

namespace srf::internal::data_plane {

Instance::Instance(std::shared_ptr<resources::PartitionResources> resources) : m_resources(std::move(resources)) {}

Instance::~Instance()
{
    call_in_destructor();
}

Client& Instance::client() const
{
    CHECK(m_client);
    return *m_client;
}

Server& Instance::server() const
{
    CHECK(m_server);
    return *m_server;
}

void Instance::do_service_start()
{
    m_resources->host()
        .main()
        .enqueue([this] {
            // if the PartitionResource has a GPU, ensure the CUDA context on the main thread is active
            // before the ucx context is constructed
            if (m_resources->device())
            {
                auto device = m_resources->device()->get();
                device.activate();

                void* addr = nullptr;
                SRF_CHECK_CUDA(cudaMalloc(&addr, 1024));
                SRF_CHECK_CUDA(cudaFree(addr));
            }

            m_context = std::make_shared<ucx::Context>();
            m_server  = std::make_unique<Server>(m_context, m_resources);
            m_client  = std::make_unique<Client>(m_context, m_resources);

            m_server->service_start();
            m_client->service_start();
        })
        .get();
}

void Instance::do_service_await_live()
{
    client().service_await_live();
    server().service_await_live();
}

void Instance::do_service_stop()
{
    client().service_stop();
    server().service_stop();
}

void Instance::do_service_kill()
{
    client().service_kill();
    server().service_kill();
}

void Instance::do_service_await_join()
{
    client().service_await_join();
    server().service_await_join();
}

}  // namespace srf::internal::data_plane

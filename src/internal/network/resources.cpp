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

#include "internal/network/resources.hpp"

#include "internal/data_plane/resources.hpp"
#include "internal/data_plane/server.hpp"
#include "internal/memory/host_resources.hpp"
#include "internal/resources/forward.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/ucx/registration_cache.hpp"
#include "internal/ucx/resources.hpp"

namespace srf::internal::network {

Resources::Resources(resources::PartitionResourceBase& base, ucx::Resources& ucx, memory::HostResources& host) :
  resources::PartitionResourceBase(base)
{
    // construct resources on the srf_network task queue thread
    ucx.network_task_queue()
        .enqueue([this, &base, &ucx, &host] {
            // initialize data plane services - server / client
            m_data_plane = std::make_unique<data_plane::Resources>(base, ucx, host);
        })
        .get();
}

Resources::~Resources()
{
    if (m_data_plane)
    {
        m_data_plane->service_stop();
        m_data_plane->service_await_join();
    }
}

data_plane::Resources& Resources::data_plane()
{
    CHECK(m_data_plane);
    return *m_data_plane;
}
}  // namespace srf::internal::network

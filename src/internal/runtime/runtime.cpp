/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/runtime/runtime.hpp"

#include "internal/remote_descriptor/manager.hpp"

namespace srf::internal::runtime {
Runtime::Runtime(resources::PartitionResources& resources) : m_resources(resources)
{
    if (resources.network())
    {
        m_remote_descriptor_manager =
            std::make_shared<remote_descriptor::Manager>(resources.network()->instance_id(), resources);
    }
}

Runtime::~Runtime()
{
    if (m_remote_descriptor_manager)
    {
        m_remote_descriptor_manager->service_stop();
        m_remote_descriptor_manager->service_await_join();
    }
}

resources::PartitionResources& Runtime::resources()
{
    return m_resources;
}
remote_descriptor::Manager& Runtime::remote_descriptor_manager()
{
    CHECK(m_remote_descriptor_manager);
    return *m_remote_descriptor_manager;
}
}  // namespace srf::internal::runtime

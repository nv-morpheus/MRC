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

#pragma once

#include "internal/remote_descriptor/remote_descriptor.hpp"
#include "internal/resources/forward.hpp"
#include "internal/resources/manager.hpp"

#include "srf/utils/macros.hpp"

#include <iterator>
#include <memory>

namespace srf::internal::runtime {

class Runtime
{
  public:
    Runtime(resources::PartitionResources& resources) : m_resources(resources)
    {
        if (resources.network())
        {
            m_remote_descriptor_manager =
                std::make_shared<remote_descriptor::Manager>(resources.network()->instance_id(), resources);
        }
    }

    ~Runtime()
    {
        if (m_remote_descriptor_manager)
        {
            m_remote_descriptor_manager->service_stop();
            m_remote_descriptor_manager->service_await_join();
        }
    }

    DELETE_COPYABILITY(Runtime);
    DELETE_MOVEABILITY(Runtime);

    resources::PartitionResources& resources()
    {
        return m_resources;
    }

    remote_descriptor::Manager& remote_descriptor_manager()
    {
        CHECK(m_remote_descriptor_manager);
        return *m_remote_descriptor_manager;
    }

  private:
    resources::PartitionResources& m_resources;
    std::shared_ptr<remote_descriptor::Manager> m_remote_descriptor_manager;
};

class RuntimeManager
{
  public:
    RuntimeManager(std::unique_ptr<resources::Manager> resources) : m_resources(std::move(resources))
    {
        CHECK(m_resources);
        for (int i = 0; i < m_resources->partition_count(); i++)
        {
            m_partitions.push_back(std::make_unique<Runtime>(m_resources->partition(i)));
        }
    }

    resources::Manager& resources()
    {
        CHECK(m_resources);
        return *m_resources;
    }

    Runtime& runtime(std::size_t partition_id)
    {
        DCHECK_LT(partition_id, m_resources->partition_count());
        DCHECK(m_partitions.at(partition_id));
        return *m_partitions.at(partition_id);
    }

  private:
    std::unique_ptr<resources::Manager> m_resources;
    std::vector<std::unique_ptr<Runtime>> m_partitions;
};

}  // namespace srf::internal::runtime

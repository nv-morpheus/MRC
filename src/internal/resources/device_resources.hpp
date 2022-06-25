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

#include "internal/resources/partition_resources_base.hpp"
#include "internal/system/device_partition.hpp"

#include <srf/memory/adaptors.hpp>

#include <memory>

namespace srf::internal::resources {

class DeviceResources : public resources::PartitionResourceBase
{
  public:
    DeviceResources(resources::PartitionResourceBase& base) : resources::PartitionResourceBase(base)
    {
        // runnable().main().enqueue([this] {
        //     m_raw = std::make_shared<memory::rmm_adaptor>()
        // }).get()
    }

    int cuda_device_id() const;

  private:
    std::shared_ptr<srf::memory::rmm_adaptor> m_raw;
    std::shared_ptr<srf::memory::rmm_adaptor> m_registered;
    std::shared_ptr<srf::memory::rmm_adaptor> m_arena;
};

}  // namespace srf::internal::resources

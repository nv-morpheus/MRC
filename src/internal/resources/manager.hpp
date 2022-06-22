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

#include "internal/resources/partition_resources.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/system/resources.hpp"
#include "internal/system/system_provider.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace srf::internal::resources {

class Manager final : public system::SystemProvider
{
  public:
    Manager(const system::SystemProvider& system);
    Manager(std::unique_ptr<system::Resources> resources);

    std::size_t device_count() const;
    std::size_t partition_count() const;

    PartitionResources& partition(std::size_t partition_id);

  private:
    std::unique_ptr<system::Resources> m_system;
    std::vector<runnable::Resources> m_runnable;
    std::vector<PartitionResources> m_partitions;
};

}  // namespace srf::internal::resources

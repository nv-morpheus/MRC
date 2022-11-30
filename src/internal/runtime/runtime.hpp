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

#include "internal/resources/manager.hpp"
#include "internal/runtime/partition.hpp"

#include "mrc/runtime/api.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace mrc::internal::runtime {

/**
 * @brief Implements the public Runtime interface and owns any high-level runtime resources, e.g. the remote descriptor
 * manager which are built on partition resources. The Runtime object is responsible for bringing up and tearing down
 * core resources manager.
 */
class Runtime final : public mrc::runtime::IRuntime
{
  public:
    Runtime(std::unique_ptr<resources::Manager> resources);
    ~Runtime() override;

    // IRuntime - total number of partitions
    std::size_t partition_count() const final;

    // IRuntime - total number of gpus / gpu partitions
    std::size_t gpu_count() const final;

    // access the partition specific resources for a given partition_id
    Partition& partition(std::size_t partition_id) final;

    // access the full set of internal resources
    resources::Manager& resources() const;

  private:
    std::unique_ptr<resources::Manager> m_resources;
    std::vector<std::unique_ptr<Partition>> m_partitions;
};

}  // namespace mrc::internal::runtime

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/memory/buffer.hpp"

#include <cstddef>
#include <memory>
#include <optional>

namespace mrc::memory {
struct memory_resource;
}  // namespace mrc::memory

namespace mrc::ucx {
class UcxResources;
}

namespace mrc::memory {

/**
 * @brief Object that provides access to device memory_resource objects for a "flattened" partition
 */
class DeviceResources : private resources::PartitionResourceBase
{
  public:
    DeviceResources(resources::PartitionResourceBase& base, std::optional<ucx::UcxResources>& ucx);

    int cuda_device_id() const;

    mrc::memory::buffer make_buffer(std::size_t bytes);

    std::shared_ptr<mrc::memory::memory_resource> system_memory_resource() const;
    std::shared_ptr<mrc::memory::memory_resource> registered_memory_resource() const;
    std::shared_ptr<mrc::memory::memory_resource> arena_memory_resource() const;

  private:
    std::shared_ptr<mrc::memory::memory_resource> m_system;
    std::shared_ptr<mrc::memory::memory_resource> m_registered;
    std::shared_ptr<mrc::memory::memory_resource> m_arena;
};

}  // namespace mrc::memory

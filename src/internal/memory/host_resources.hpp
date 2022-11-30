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

#include "internal/runnable/resources.hpp"
#include "internal/system/host_partition_provider.hpp"
#include "internal/ucx/registation_callback_builder.hpp"

#include "mrc/memory/buffer.hpp"
#include "mrc/memory/resources/memory_resource.hpp"

#include <cstddef>
#include <memory>

namespace mrc::internal::memory {

/**
 * @brief Object that provides access to host memory_resource objects for a given host partition
 */
class HostResources final : private system::HostPartitionProvider
{
  public:
    HostResources(runnable::Resources& runnable, ucx::RegistrationCallbackBuilder&& callbacks);

    mrc::memory::buffer make_buffer(std::size_t bytes);

    std::shared_ptr<mrc::memory::memory_resource> system_memory_resource();
    std::shared_ptr<mrc::memory::memory_resource> registered_memory_resource();
    std::shared_ptr<mrc::memory::memory_resource> arena_memory_resource();

  private:
    std::shared_ptr<mrc::memory::memory_resource> m_system;
    std::shared_ptr<mrc::memory::memory_resource> m_registered;
    std::shared_ptr<mrc::memory::memory_resource> m_arena;
};

}  // namespace mrc::internal::memory

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

#include "internal/memory/block_manager.hpp"
#include "internal/memory/ucx_memory_block.hpp"
#include "internal/memory/ucx_registered_resource.hpp"
#include "internal/resources/runnable_provider.hpp"
#include "internal/ucx/resources.hpp"

#include <srf/memory/resources/device/cuda_malloc_resource.hpp>
#include <srf/memory/resources/host/malloc_memory_resource.hpp>
#include <srf/memory/resources/host/pinned_memory_resource.hpp>
#include <srf/memory/resources/memory_resource.hpp>

#include <memory>

namespace srf::internal::memory {

class Resources final : public resources::PartitionResourceBase
{
  public:
    Resources(resources::PartitionResourceBase& base, std::optional<ucx::Resources>& ucx_resources);

  private:
    std::unique_ptr<srf::memory::memory_resource> m_host_raw;
    std::unique_ptr<srf::memory::memory_resource> m_device_raw;

    std::shared_ptr<UcxRegistrationCache> m_host_reg_cache;
    std::shared_ptr<UcxRegistrationCache> m_device_reg_cache;

    std::unique_ptr<srf::memory::memory_resource> m_host_resource;
    std::unique_ptr<srf::memory::memory_resource> m_device_resource;
};

}  // namespace srf::internal::memory

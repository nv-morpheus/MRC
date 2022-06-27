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

#include "internal/resources/device_resources.hpp"

#include "internal/ucx/resources.hpp"

#include "srf/memory/resources/arena_resource.hpp"
#include "srf/memory/resources/device/cuda_malloc_resource.hpp"
#include "srf/memory/resources/logging_resource.hpp"

#include <glog/logging.h>

#include <sstream>
#include <utility>

namespace srf::internal::resources {

DeviceResources::DeviceResources(runnable::Resources& runnable,
                                 std::size_t partition_id,
                                 std::optional<ucx::Resources>& ucx) :
  resources::PartitionResourceBase(runnable, partition_id)
{
    CHECK(partition().has_device());

    runnable.main()
        .enqueue([this, &ucx] {
            std::stringstream device_prefix;
            device_prefix << "cuda_malloc:" << cuda_device_id();

            auto cuda_malloc = std::make_unique<srf::memory::cuda_malloc_resource>(cuda_device_id());
            m_system         = srf::memory::make_shared_resource<srf::memory::logging_resource>(std::move(cuda_malloc),
                                                                                        device_prefix.str());

            if (ucx)
            {
                m_registered = ucx->adapt_to_registered_resource(m_system);
            }
            else
            {
                m_registered = m_system;
            }

            if (system().options().resources().enable_device_memory_pool())
            {
                const auto& opts = system().options().resources().device_memory_pool();

                VLOG(10) << "partition_id: " << this->partition_id()
                         << " constructing arena memory_resource with initial=" << bytes_to_string(opts.block_size())
                         << "; max bytes=" << bytes_to_string(opts.max_aggreate_bytes());

                m_arena = srf::memory::make_shared_resource<srf::memory::arena_resource>(
                    m_registered, opts.block_size(), opts.max_aggreate_bytes());
            }
            else
            {
                m_arena = m_registered;
            }
        })
        .get();
}

int DeviceResources::cuda_device_id() const
{
    return partition().device().cuda_device_id();
}

}  // namespace srf::internal::resources

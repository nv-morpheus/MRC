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

#include "internal/memory/device_resources.hpp"

#include "internal/system/device_partition.hpp"
#include "internal/system/partition.hpp"
#include "internal/system/system.hpp"
#include "internal/ucx/resources.hpp"

#include "srf/core/task_queue.hpp"
#include "srf/cuda/device_guard.hpp"
#include "srf/memory/adaptors.hpp"
#include "srf/memory/resources/arena_resource.hpp"
#include "srf/memory/resources/device/cuda_malloc_resource.hpp"
#include "srf/memory/resources/logging_resource.hpp"
#include "srf/options/options.hpp"
#include "srf/options/resources.hpp"
#include "srf/types.hpp"
#include "srf/utils/bytes_to_string.hpp"

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace srf::internal::memory {

DeviceResources::DeviceResources(resources::PartitionResourceBase& base, std::optional<ucx::Resources>& ucx) :
  resources::PartitionResourceBase(base)
{
    CHECK(partition().has_device());

    runnable()
        .main()
        .enqueue([this, &ucx] {
            std::stringstream device_prefix;
            device_prefix << "cuda_malloc:" << cuda_device_id();

            DeviceGuard guard(cuda_device_id());

            auto cuda_malloc = std::make_unique<srf::memory::cuda_malloc_resource>(cuda_device_id());
            m_system         = srf::memory::make_shared_resource<srf::memory::logging_resource>(std::move(cuda_malloc),
                                                                                        device_prefix.str());

            if (ucx)
            {
                m_registered = ucx->adapt_to_registered_resource(m_system, cuda_device_id());
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

srf::memory::buffer DeviceResources::make_buffer(std::size_t bytes)
{
    return srf::memory::buffer(bytes, m_arena);
}
}  // namespace srf::internal::memory

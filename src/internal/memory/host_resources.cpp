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

#include "internal/memory/host_resources.hpp"

#include "internal/memory/callback_adaptor.hpp"
#include "internal/system/host_partition.hpp"
#include "internal/system/system.hpp"

#include "srf/core/task_queue.hpp"
#include "srf/memory/adaptors.hpp"
#include "srf/memory/resources/arena_resource.hpp"
#include "srf/memory/resources/host/malloc_memory_resource.hpp"
#include "srf/memory/resources/host/pinned_memory_resource.hpp"
#include "srf/memory/resources/logging_resource.hpp"
#include "srf/memory/resources/memory_resource.hpp"
#include "srf/options/options.hpp"
#include "srf/options/resources.hpp"
#include "srf/types.hpp"
#include "srf/utils/bytes_to_string.hpp"

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace srf::internal::memory {

HostResources::HostResources(runnable::Resources& runnable, ucx::RegistrationCallbackBuilder&& callbacks) :
  system::HostPartitionProvider(runnable)
{
    runnable.main()
        .enqueue([this, &callbacks] {
            // logging prefix
            std::stringstream prefix;

            // construct raw memory_resource from malloc or pinned if device(s) present
            if (host_partition().device_partition_ids().empty())
            {
                m_system = std::make_shared<srf::memory::malloc_memory_resource>();
                prefix << "malloc";
            }
            else
            {
                m_system = std::make_shared<srf::memory::pinned_memory_resource>();
                prefix << "cuda_pinned";
            }

            prefix << ":" << host_partition_id();
            m_system =
                srf::memory::make_shared_resource<srf::memory::logging_resource>(std::move(m_system), prefix.str());

            // adapt to callback resource if we have callbacks
            if (callbacks.size() == 0)
            {
                m_registered = m_system;
            }
            else
            {
                m_registered =
                    srf::memory::make_shared_resource<memory::CallbackAdaptor>(m_system, std::move(callbacks));
            }

            // adapt to arena
            if (system().options().resources().enable_host_memory_pool())
            {
                const auto& opts = system().options().resources().host_memory_pool();

                VLOG(10) << "host_partition_id: " << host_partition_id()
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

srf::memory::buffer HostResources::make_buffer(std::size_t bytes)
{
    return srf::memory::buffer(bytes, m_arena);
}
}  // namespace srf::internal::memory

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

#include "mrc/core/task_queue.hpp"
#include "mrc/memory/adaptors.hpp"
#include "mrc/memory/resources/arena_resource.hpp"
#include "mrc/memory/resources/host/malloc_memory_resource.hpp"
#include "mrc/memory/resources/host/pinned_memory_resource.hpp"
#include "mrc/memory/resources/logging_resource.hpp"
#include "mrc/memory/resources/memory_resource.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/resources.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/bytes_to_string.hpp"

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

namespace mrc::internal::memory {

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
                m_system = std::make_shared<mrc::memory::malloc_memory_resource>();
                prefix << "malloc";
            }
            else
            {
                m_system = std::make_shared<mrc::memory::pinned_memory_resource>();
                prefix << "cuda_pinned";
            }

            prefix << ":" << host_partition_id();
            m_system =
                mrc::memory::make_shared_resource<mrc::memory::logging_resource>(std::move(m_system), prefix.str());

            // adapt to callback resource if we have callbacks
            if (callbacks.size() == 0)
            {
                m_registered = m_system;
            }
            else
            {
                m_registered =
                    mrc::memory::make_shared_resource<memory::CallbackAdaptor>(m_system, std::move(callbacks));
            }

            // adapt to arena
            if (system().options().resources().enable_host_memory_pool())
            {
                const auto& opts = system().options().resources().host_memory_pool();

                VLOG(10) << "host_partition_id: " << host_partition_id()
                         << " constructing arena memory_resource with initial=" << bytes_to_string(opts.block_size())
                         << "; max bytes=" << bytes_to_string(opts.max_aggreate_bytes());

                m_arena = mrc::memory::make_shared_resource<mrc::memory::arena_resource>(
                    m_registered, opts.block_size(), opts.max_aggreate_bytes());
            }
            else
            {
                m_arena = m_registered;
            }
        })
        .get();
}

mrc::memory::buffer HostResources::make_buffer(std::size_t bytes)
{
    return mrc::memory::buffer(bytes, m_arena);
}
std::shared_ptr<mrc::memory::memory_resource> HostResources::system_memory_resource()
{
    return m_system;
}
std::shared_ptr<mrc::memory::memory_resource> HostResources::registered_memory_resource()
{
    return m_registered;
}
std::shared_ptr<mrc::memory::memory_resource> HostResources::arena_memory_resource()
{
    return m_arena;
}
}  // namespace mrc::internal::memory

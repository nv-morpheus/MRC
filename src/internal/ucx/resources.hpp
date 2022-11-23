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

#include "internal/resources/partition_resources_base.hpp"
#include "internal/system/fiber_task_queue.hpp"
#include "internal/ucx/context.hpp"
#include "internal/ucx/endpoint.hpp"
#include "internal/ucx/registation_callback_builder.hpp"
#include "internal/ucx/registration_cache.hpp"
#include "internal/ucx/registration_resource.hpp"
#include "internal/ucx/worker.hpp"

#include "mrc/core/task_queue.hpp"
#include "mrc/memory/adaptors.hpp"
#include "mrc/runnable/launch_options.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace mrc::internal::network {
class Resources;
}

namespace mrc::internal::ucx {

/**
 * @brief UCX Resources - if networking is enabled, there should be 1 UCX Resource per "flattened" partition
 */
class Resources final : public resources::PartitionResourceBase
{
  public:
    Resources(resources::PartitionResourceBase& base, system::FiberTaskQueue& network_task_queue);

    using resources::PartitionResourceBase::partition;

    // ucx worker associated with this partitions ucx context
    Worker& worker();

    // task queue used to run the data plane's progress engine
    mrc::core::FiberTaskQueue& network_task_queue();

    // registration cache to look up local/remote keys for registered blocks of memory
    RegistrationCache& registration_cache();

    // used to build a callback adaptor memory resource for host memory resources
    void add_registration_cache_to_builder(RegistrationCallbackBuilder& builder);

    // used to build device memory resources that are registered with the ucx context
    template <typename UpstreamT>
    auto adapt_to_registered_resource(UpstreamT upstream, int cuda_device_id)
    {
        return mrc::memory::make_unique_resource<RegistrationResource>(
            std::move(upstream), m_registration_cache, cuda_device_id);
    }

    std::shared_ptr<ucx::Endpoint> make_ep(const std::string& worker_address) const;

    static mrc::runnable::LaunchOptions launch_options(std::uint64_t concurrency);

  private:
    system::FiberTaskQueue& m_network_task_queue;
    std::shared_ptr<Context> m_ucx_context;
    std::shared_ptr<Worker> m_worker;
    std::shared_ptr<RegistrationCache> m_registration_cache;

    // enable direct access to context and workers
    friend network::Resources;
};

}  // namespace mrc::internal::ucx

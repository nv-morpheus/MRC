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
#include "internal/runnable/resources.hpp"
#include "internal/system/fiber_task_queue.hpp"
#include "internal/ucx/context.hpp"
#include "internal/ucx/registation_callback_builder.hpp"
#include "internal/ucx/registration_cache.hpp"
#include "internal/ucx/registration_resource.hpp"
#include "internal/ucx/worker.hpp"

#include "srf/memory/adaptors.hpp"

#include <glog/logging.h>

#include <cstddef>
#include <memory>

namespace srf::internal::network {
class Resources;
}

namespace srf::internal::ucx {

class Resources final : private resources::PartitionResourceBase
{
  public:
    Resources(runnable::Resources& _runnable_resources,
              std::size_t _partition_id,
              system::FiberTaskQueue& network_task_queue);

    using resources::PartitionResourceBase::partition;

    srf::core::FiberTaskQueue& network_task_queue();
    const RegistrationCache& registration_cache() const;

    void add_registration_cache_to_builder(RegistrationCallbackBuilder& builder);

    template <typename UpstreamT>
    auto adapt_to_registered_resource(UpstreamT upstream, int cuda_device_id)
    {
        return srf::memory::make_unique_resource<RegistrationResource>(
            std::move(upstream), m_registration_cache, cuda_device_id);
    }

  private:
    system::FiberTaskQueue& m_network_task_queue;
    std::shared_ptr<Context> m_ucx_context;
    std::shared_ptr<Worker> m_worker_server;
    std::shared_ptr<Worker> m_worker_client;
    std::shared_ptr<RegistrationCache> m_registration_cache;

    // enable direct access to context and workers
    friend network::Resources;
};

}  // namespace srf::internal::ucx

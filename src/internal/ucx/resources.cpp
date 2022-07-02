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

#include "internal/ucx/resources.hpp"

#include "internal/resources/partition_resources_base.hpp"
#include "internal/system/device_partition.hpp"
#include "internal/system/fiber_task_queue.hpp"
#include "internal/system/partition.hpp"
#include "internal/ucx/endpoint.hpp"
#include "internal/ucx/worker.hpp"

#include "srf/core/task_queue.hpp"
#include "srf/cuda/common.hpp"
#include "srf/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <cuda_runtime.h>

#include <ostream>

namespace srf::internal::ucx {

Resources::Resources(resources::PartitionResourceBase& base, system::FiberTaskQueue& network_task_queue) :
  resources::PartitionResourceBase(base),
  m_network_task_queue(network_task_queue)
{
    VLOG(1) << "constructing network resources for partition: " << partition_id() << " on partitions main task queue";
    m_network_task_queue
        .enqueue([this] {
            if (partition().has_device())
            {
                void* tmp = nullptr;
                DVLOG(10) << "partition: " << partition_id()
                          << " has a gpu present; ensure a cuda context is active before instantiating a ucx context";
                SRF_CHECK_CUDA(cudaSetDevice(partition().device().cuda_device_id()));
                SRF_CHECK_CUDA(cudaMalloc(&tmp, 1024));
                SRF_CHECK_CUDA(cudaFree(tmp));
            }

            // we need to create both the context and the workers to ensure ucx and cuda are aligned

            DVLOG(10) << "initializing ucx context";
            m_ucx_context = std::make_shared<Context>();

            DVLOG(10) << "initialize a ucx data_plane worker";
            m_worker = std::make_shared<Worker>(m_ucx_context);

            DVLOG(10) << "initialize the registration cache for this context";
            m_registration_cache = std::make_shared<RegistrationCache>(m_ucx_context);

            // flush any work that needs to be done by the workers
            while (m_worker->progress() != 0) {}
        })
        .get();
}

void Resources::add_registration_cache_to_builder(RegistrationCallbackBuilder& builder)
{
    builder.add_registration_cache(m_registration_cache);
}

srf::core::FiberTaskQueue& Resources::network_task_queue()
{
    return m_network_task_queue;
}
const RegistrationCache& Resources::registration_cache() const
{
    CHECK(m_registration_cache);
    return *m_registration_cache;
}
Worker& Resources::worker()
{
    CHECK(m_worker);
    return *m_worker;
}
std::shared_ptr<ucx::Endpoint> Resources::make_ep(const std::string& worker_address) const
{
    return std::make_shared<ucx::Endpoint>(m_worker, worker_address);
}

}  // namespace srf::internal::ucx

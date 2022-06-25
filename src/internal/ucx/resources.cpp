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

#include "internal/system/system_provider.hpp"
#include "internal/ucx/worker.hpp"

#include "srf/cuda/common.hpp"

namespace srf::internal::ucx {

Resources::Resources(runnable::Resources& _runnable_resources, std::size_t _partition_id) :
  resources::PartitionResourceBase(_runnable_resources, _partition_id)
{
    VLOG(1) << "constructing network resources for partition: " << partition_id() << " on partitions main task queue";
    runnable()
        .main()
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

            DVLOG(10) << "initializing ucx context";
            m_ucx_context = std::make_shared<Context>();

            DVLOG(10) << "initialize a ucx data_plane worker for server";
            m_worker_server = std::make_shared<Worker>(m_ucx_context);

            DVLOG(10) << "initialize a ucx data_plane worker for client";
            m_worker_client = std::make_shared<Worker>(m_ucx_context);

            DVLOG(10) << "initialize the registration cache for this context";
            m_registration_cache = std::make_shared<RegistrationCache>(m_ucx_context);

            // flush any work that needs to be done by the workers
            while (m_worker_server->progress() != 0) {}
            while (m_worker_client->progress() != 0) {}
        })
        .get();
}

Context& Resources::context()
{
    CHECK(m_ucx_context);
    return *m_ucx_context;
}

void Resources::add_registration_cache_to_builder(RegistrationCallbackBuilder& builder)
{
    builder.add_registration_cache(m_registration_cache);
}
}  // namespace srf::internal::ucx

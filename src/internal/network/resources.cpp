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

#include "internal/network/resources.hpp"
#include "internal/system/system_provider.hpp"

namespace srf::internal::network {

Resources::Resources(const resources::RunnableProvider& partition_provider) :
  resources::RunnableProvider(partition_provider)
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
            m_ucx_context = std::make_shared<ucx::Context>();

            DVLOG(10) << "initialize a ucx data_plane server to handle incoming events";
            m_server = std::make_unique<data_plane::Server>(*this, std::make_shared<ucx::Worker>(m_ucx_context));

            DVLOG(10) << "initialize a ucx data_plane client to make network requests";
            DVLOG(10) << "initialize a grpc control_plane client to make network requests";
        })
        .get();
}

}  // namespace srf::internal::network

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

namespace srf::internal::network {
Resources::Resources(runnable::Resources& runnable_resources) : m_runnable(runnable_resources)
{
    VLOG(1) << "constructing network resources for partition: " << m_runnable.partition_id()
            << " on partitions main task queue";
    m_runnable.main()
        .enqueue([this] {
            if (m_runnable.partition().has_device())
            {
                void* tmp = nullptr;
                VLOG(10) << "partition: " << m_runnable.partition_id()
                         << " has a gpu present; ensure a cuda context is active before instantiating a ucx context";
                SRF_CHECK_CUDA(cudaSetDevice(m_runnable.partition().device().cuda_device_id()));
                SRF_CHECK_CUDA(cudaMalloc(&tmp, 1024));
                SRF_CHECK_CUDA(cudaFree(tmp));
            }

            VLOG(10) << "initializing ucx context";
            m_ucx_context = std::make_shared<ucx::Context>();

            VLOG(10) << "initialize a ucx data_plane server to handle incoming events";
            m_server = std::make_unique<data_plane::Server>(m_ucx_context, m_runnable);

            VLOG(10) << "initialize a ucx data_plane client to make network requests";
            VLOG(10) << "initialize a grpc control_plane client to make network requests";
        })
        .get();
}
}  // namespace srf::internal::network

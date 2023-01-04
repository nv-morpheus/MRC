/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/cuda/device_guard.hpp"

#include "mrc/cuda/common.hpp"

#include <cuda_runtime.h>
#include <glog/logging.h>
// IWYU thinks we need std::allocator for the debug macros (and only in debug builds)
// IWYU pragma: no_include <memory>

namespace mrc {

DeviceGuard::DeviceGuard(int device_id)
{
    DCHECK_GE(device_id, 0);
    MRC_CHECK_CUDA(cudaGetDevice(&m_DeviceID));
    MRC_CHECK_CUDA(cudaSetDevice(device_id));
}

DeviceGuard::~DeviceGuard()
{
    MRC_CHECK_CUDA(cudaSetDevice(m_DeviceID));
}

}  // namespace mrc

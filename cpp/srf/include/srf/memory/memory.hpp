/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <srf/memory/memory_kind.hpp>
#include <srf/memory/resource_view.hpp>

#include <cuda/memory_resource>

namespace srf::memory {

using HostResourceView   = resource_view<::cuda::memory_access::host,  // NOLINT
                                       ::cuda::memory_access::device,
                                       ::cuda::memory_location::host,
                                       ::cuda::resident>;
using DeviceResourceView =  // NOLINT
    resource_view<::cuda::memory_access::device, ::cuda::memory_location::device, ::cuda::resident>;

}  // namespace srf::memory

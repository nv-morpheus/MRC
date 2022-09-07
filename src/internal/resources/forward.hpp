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

#pragma once

namespace srf::internal {

namespace resources {
class Manager;
class PartitionResourceBase;
}  // namespace resources

namespace runnable {
class Resources;
}  // namespace runnable

namespace memory {
class HostResources;
class DeviceResources;
}  // namespace memory

// control plane and data plane
namespace network {
class Resources;
}  // namespace network

namespace ucx {
class Resources;
}  // namespace ucx

namespace data_plane {
class Resources;
}  // namespace data_plane

}  // namespace srf::internal

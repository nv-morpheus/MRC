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

namespace mrc::core {

/**
 * @brief Public Runtime Interface
 *
 * Any execution context can query its unique runtime by calling `mrc::runtime()`. This will return a Runtime object
 * that is specialized to execution context of the caller. The caller has access to all of the resource partition on
 * the physical machine, however the Partition assigned the current execution context by the scheduler is provided by
 * the Partition methods - host(), device(int=0), and device_count(). The current execution should primary uses these
 * partitions for host and device(s) allocation and kernel launches.
 */
class Runtime  // : public Partition
{
  public:
    virtual ~Runtime() = default;
    // ~Runtime() override = default;

    // virtual std::size_t host_partitions_count()   = 0;
    // virtual std::size_t device_partitions_count() = 0;

    // virtual const HostPartition& host_partitions(const std::uint32_t&) const     = 0;
    // virtual const DevicePartition& device_partitions(const std::uint32_t&) const = 0;
};

}  // namespace mrc::core

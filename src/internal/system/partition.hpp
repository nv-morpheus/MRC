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

#include "internal/system/device_partition.hpp"
#include "internal/system/host_partition.hpp"

#include <cstddef>
#include <memory>

namespace mrc::internal::system {

class Partition final
{
  public:
    Partition(std::size_t host_partition_id,
              std::shared_ptr<const HostPartition> host,
              std::shared_ptr<const DevicePartition> device);

    const HostPartition& host() const;
    const DevicePartition& device() const;

    size_t host_partition_id() const;
    bool has_device() const;

  private:
    std::size_t m_host_partition_id;
    std::shared_ptr<const HostPartition> m_host;
    std::shared_ptr<const DevicePartition> m_device;
};

}  // namespace mrc::internal::system

/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/runtime/partition_manager.hpp"

#include "internal/runnable/resources.hpp"
#include "internal/system/partition.hpp"

#include <glog/logging.h>

namespace mrc::internal::runtime {

PartitionManager::PartitionManager(const resources::PartitionResources& resources,
                                   control_plane::Client& control_plane_client) :
  m_resources(resources),
  m_control_plane_client(control_plane_client)
{}

PartitionManager::~PartitionManager() = default;

}  // namespace mrc::internal::runtime

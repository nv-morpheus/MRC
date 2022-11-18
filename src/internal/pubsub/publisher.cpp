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

#include "internal/pubsub/publisher.hpp"

#include "internal/codable/codable_storage.hpp"
#include "internal/control_plane/client.hpp"
#include "internal/resources/partition_resources.hpp"

namespace srf::internal::pubsub {

Publisher::Publisher(std::string service_name, resources::PartitionResources& resources) :
  control_plane::client::SubscriptionService(std::move(service_name), resources.network()->control_plane()),
  m_resources(resources)
{}

std::unique_ptr<srf::codable::ICodableStorage> Publisher::create_storage()
{
    return std::make_unique<codable::CodableStorage>(m_resources);
}

void Publisher::stop()
{
    this->release_channel();
}
bool Publisher::is_live() const
{
    return this->has_channel();
}
void Publisher::await_join()
{
    this->release_channel();
}
}  // namespace srf::internal::pubsub

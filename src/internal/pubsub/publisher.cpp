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

namespace srf::internal::pubsub {

Publisher::Publisher(std::string service_name, std::uint64_t tag, resources::PartitionResources& resources) :
  m_service_name(std::move(service_name)),
  m_tag(tag),
  m_resources(resources)
{}

const std::string& Publisher::service_name() const
{
    return m_service_name;
}

const std::uint64_t& Publisher::tag() const
{
    return m_tag;
}

std::unique_ptr<srf::codable::ICodableStorage> Publisher::create_storage()
{
    return std::make_unique<codable::CodableStorage>(m_resources);
}

}  // namespace srf::internal::pubsub

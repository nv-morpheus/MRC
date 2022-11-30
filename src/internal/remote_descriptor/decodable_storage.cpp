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

#include "internal/remote_descriptor/decodable_storage.hpp"

#include <utility>

namespace mrc::internal::remote_descriptor {

DecodableStorage::DecodableStorage(mrc::codable::protos::RemoteDescriptor&& proto,
                                   resources::PartitionResources& resources) :
  m_proto(std::move(proto)),
  m_resources(resources)
{}

const mrc::codable::protos::EncodedObject& DecodableStorage::get_proto() const
{
    return m_proto.encoded_object();
}

resources::PartitionResources& DecodableStorage::resources() const
{
    return m_resources;
}

const mrc::codable::protos::RemoteDescriptor& DecodableStorage::remote_descriptor_proto() const
{
    return m_proto;
}
}  // namespace mrc::internal::remote_descriptor

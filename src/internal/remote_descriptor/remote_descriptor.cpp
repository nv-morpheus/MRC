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

#include "internal/remote_descriptor/remote_descriptor.hpp"

#include "internal/remote_descriptor/manager.hpp"
#include "internal/resources/partition_resources.hpp"

namespace srf::internal::remote_descriptor {

RemoteDescriptor::RemoteDescriptor(std::shared_ptr<Manager> manager,
                                   std::unique_ptr<srf::codable::protos::RemoteDescriptor> rd,
                                   resources::PartitionResources& resources) :
  m_manager(std::move(manager)),
  m_descriptor(std::move(rd)),
  m_encoded_object(std::make_unique<EncodedObject>(m_descriptor->encoded_object(), resources))
{}

RemoteDescriptor::~RemoteDescriptor()
{
    release();
}

void RemoteDescriptor::release()
{
    if (m_descriptor)
    {
        CHECK(m_manager);
        m_manager->decrement_tokens(std::move(m_descriptor));
        m_manager.reset();
        m_encoded_object.reset();
    }
}

RemoteDescriptor::operator bool() const
{
    return bool(m_descriptor);
}

std::unique_ptr<const srf::codable::protos::RemoteDescriptor> RemoteDescriptor::release_ownership()
{
    m_manager.reset();
    m_encoded_object.reset();
    return std::move(m_descriptor);
}

const EncodedObject& RemoteDescriptor::encoded_object() const
{
    CHECK(m_encoded_object);
    return *m_encoded_object;
}
}  // namespace srf::internal::remote_descriptor

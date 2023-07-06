/*
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

#include "mrc/runtime/remote_descriptor.hpp"

#include "internal/codable/codable_storage.hpp"
#include "internal/resources/system_resources.hpp"

#include "mrc/runtime/remote_descriptor_handle.hpp"
#include "mrc/runtime/remote_descriptor_manager.hpp"

#include <memory>
#include <utility>

namespace mrc::runtime {

// Save the descriptor from the thread local value at the time of creation
Descriptor::Descriptor() : m_partition_resources(resources::SystemResources::get_partition()) {}

Descriptor::Descriptor(Descriptor&& other) = default;

Descriptor& Descriptor::operator=(Descriptor&& other) = default;
// Descriptor& Descriptor::operator=(Descriptor&& other)
// {
//     m_partition_resources = std::move(other.m_partition_resources);

//     return *this;
// }

std::unique_ptr<mrc::codable::ICodableStorage> Descriptor::make_storage() const
{
    return std::make_unique<codable::CodableStorage>(m_partition_resources);
}

RemoteDescriptor::RemoteDescriptor() = default;

RemoteDescriptor::RemoteDescriptor(std::unique_ptr<codable::IDecodableStorage> storage) : m_storage(std::move(storage))
{}

RemoteDescriptor::RemoteDescriptor(RemoteDescriptor&& other) noexcept = default;

RemoteDescriptor& RemoteDescriptor::operator=(RemoteDescriptor&& other) noexcept = default;

RemoteDescriptor::~RemoteDescriptor() = default;

bool RemoteDescriptor::has_value() const
{
    return bool(m_storage);
}

std::unique_ptr<codable::IDecodableStorage> RemoteDescriptor::release_storage()
{
    CHECK(this->has_value()) << "Cannot get a storage from a Descriptor which has been released or transferred.";

    return std::move(m_storage);
}

// bool RemoteDescriptor::has_value() const
// {
//     return (m_manager && m_handle);
// }

// void RemoteDescriptor::release_ownership()
// {
//     if (m_manager)
//     {
//         CHECK(m_handle);
//         m_manager->release_handle(std::move(m_handle));
//         m_manager.reset();
//     }
// }

// std::unique_ptr<IRemoteDescriptorHandle> RemoteDescriptor::release_handle()
// {
//     CHECK(*this);
//     m_manager.reset();
//     return std::move(m_handle);
// }

// RemoteDescriptor::operator bool() const
// {
//     return has_value();
// }

}  // namespace mrc::runtime

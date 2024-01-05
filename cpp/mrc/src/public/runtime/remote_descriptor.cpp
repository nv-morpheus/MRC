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
#include "internal/data_plane/data_plane_resources.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/ucx/memory_block.hpp"
#include "internal/ucx/registration_cache.hpp"

#include "mrc/memory/memory_block_provider.hpp"
#include "mrc/protos/codable.pb.h"

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

codable::LocalSerializedWrapper& LocalDescriptor2::encoded_object() const
{
    return *m_encoded_object;
}

std::unique_ptr<LocalDescriptor2> LocalDescriptor2::from_value(
    std::unique_ptr<ValueDescriptor> value_descriptor,
    std::shared_ptr<memory::memory_block_provider> block_provider)
{
    // Create a wrapper around the memory block provider to track the created memory blocks
    // auto wrapper = std::make_shared<MemoryBlockProviderWrapper>(block_provider);
    auto wrapper = block_provider;

    // Serialize the object
    auto encoded_object = value_descriptor->encode(wrapper);

    return std::unique_ptr<LocalDescriptor2>(
        new LocalDescriptor2(std::move(encoded_object), std::move(value_descriptor)));
}

std::unique_ptr<LocalDescriptor2> LocalDescriptor2::from_remote(std::unique_ptr<RemoteDescriptor2> remote_descriptor,
                                                                data_plane::DataPlaneResources2& data_plane_resources)
{
    return nullptr;
}

LocalDescriptor2::LocalDescriptor2(std::unique_ptr<codable::LocalSerializedWrapper> encoded_object,
                                   std::unique_ptr<ValueDescriptor> value_descriptor) :
  m_encoded_object(std::move(encoded_object)),
  m_value_descriptor(std::move(value_descriptor))
{}

RemoteDescriptor2::RemoteDescriptor2(std::unique_ptr<codable::protos::RemoteSerializedObject> encoded_object) :
  m_serialized_object(std::move(encoded_object))
{}

std::unique_ptr<RemoteDescriptor2> RemoteDescriptor2::from_local(std::unique_ptr<LocalDescriptor2> local_desc,
                                                                 data_plane::DataPlaneResources2& data_plane_resources)
{
    auto remote_object = std::make_unique<codable::protos::RemoteSerializedObject>();

    // Transfer the info object
    remote_object->set_allocated_info(local_desc->encoded_object().proto().release_info());

    // Loop over all local payloads and convert them to remote payloads

    for (const auto& local_payload : local_desc->encoded_object().proto().payloads())
    {
        auto* remote_payload = remote_object->add_payloads();

        auto ucx_block = data_plane_resources.registration_cache().lookup(local_payload.address());

        bool should_cache = false;  // Not sure what to set this

        remote_payload->set_instance_id(0);
        remote_payload->set_address(local_payload.address());
        remote_payload->set_bytes(local_payload.bytes());
        remote_payload->set_memory_block_address(reinterpret_cast<std::uint64_t>(ucx_block->data()));
        remote_payload->set_memory_block_size(ucx_block->bytes());
        remote_payload->set_memory_kind(local_payload.memory_kind());
        remote_payload->set_remote_key(ucx_block->packed_remote_keys());
        remote_payload->set_should_cache(should_cache);
    }

    return std::unique_ptr<RemoteDescriptor2>(new RemoteDescriptor2(std::move(remote_object)));
}

std::unique_ptr<RemoteDescriptor2> RemoteDescriptor2::from_encoded_object(
    std::unique_ptr<codable::LocalSerializedWrapper> encoded_object)
{
    return nullptr;
}

}  // namespace mrc::runtime

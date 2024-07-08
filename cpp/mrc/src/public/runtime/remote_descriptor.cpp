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
#include "internal/memory/host_resources.hpp"
#include "internal/remote_descriptor/messages.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/ucx/memory_block.hpp"
#include "internal/ucx/registration_cache.hpp"

#include "mrc/memory/buffer.hpp"
#include "mrc/memory/memory_block_provider.hpp"
#include "mrc/memory/resources/host/malloc_memory_resource.hpp"
#include "mrc/memory/resources/memory_resource.hpp"
#include "mrc/protos/codable.pb.h"

#include <ucp/api/ucp.h>
#include <ucxx/api.h>

#include <cstdint>
#include <limits>
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

codable::DescriptorObjectHandler& LocalDescriptor2::encoded_object() const
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
    auto encoded_object = value_descriptor->encode();

    return std::unique_ptr<LocalDescriptor2>(
        new LocalDescriptor2(std::move(encoded_object), std::move(value_descriptor)));
}

std::unique_ptr<LocalDescriptor2> LocalDescriptor2::from_remote(std::unique_ptr<RemoteDescriptor2> remote_descriptor,
                                                                data_plane::DataPlaneResources2& data_plane_resources)
{
    auto local_obj = std::make_unique<codable::DescriptorObjectHandler>();

    auto mr = memory::malloc_memory_resource::instance();

    std::vector<std::shared_ptr<ucxx::Request>> requests;
    std::vector<memory::buffer> buffers;

    // Get the endpoint of the remote descriptor
    auto ep = data_plane_resources.find_endpoint(remote_descriptor->encoded_object().instance_id());

    // Loop over all remote payloads and convert them to local payloads
    for (const auto& remote_payload : remote_descriptor->encoded_object().payloads())
    {
        auto* local_payload = local_obj->proto().add_payloads();

        // Set the payload memory kind
        local_payload->set_memory_kind(remote_payload.memory_kind());

        // If payload is an EagerMessage, we do not need to do any pulling
        if (remote_payload.has_eager_msg())
        {
            auto* eager_msg = local_payload->mutable_eager_msg();
            eager_msg->set_data(remote_payload.eager_msg().data());
            continue;
        }

        // Initialize a new DeferredMessage in the local descriptor
        auto* deferred_msg = local_payload->mutable_deferred_msg();

        // Get the DeferredMessage of the remote payload
        auto deferred_remote_msg = remote_payload.deferred_msg();

        // Allocate the memory needed for this and prevent it from going out-of-scope before request completes
        buffers.emplace_back(deferred_remote_msg.bytes(), mr);

        // now issue the request
        requests.push_back(data_plane_resources.memory_recv_async(ep,
                                                                  buffers.back(),
                                                                  deferred_remote_msg.address(),
                                                                  deferred_remote_msg.remote_key().data()));

        deferred_msg->set_address(reinterpret_cast<uintptr_t>(buffers.back().data()));
        deferred_msg->set_bytes(buffers.back().bytes());
    }

    // Now, we need to wait for all requests to be complete
    data_plane_resources.wait_requests(requests);

    // For the remote descriptor message, send decrement to the remote resources
    remote_descriptor::RemoteDescriptorDecrementMessage dec_message;
    dec_message.object_id = remote_descriptor->encoded_object().object_id();
    dec_message.tokens    = remote_descriptor->encoded_object().tokens();

    // TODO(Peter): Define `ucxx::AmReceiverCallbackInfo` at central place, must be known by all MRC processes.
    // Send a decrement message using custom AM receiver callback
    auto decrement_request = ep->amSend(&dec_message,
                                        sizeof(remote_descriptor::RemoteDescriptorDecrementMessage),
                                        UCS_MEMORY_TYPE_HOST,
                                        ucxx::AmReceiverCallbackInfo("MRC", 0));

    return std::unique_ptr<LocalDescriptor2>(new LocalDescriptor2(std::move(local_obj)));
}

LocalDescriptor2::LocalDescriptor2(std::unique_ptr<codable::DescriptorObjectHandler> encoded_object,
                                   std::unique_ptr<ValueDescriptor> value_descriptor) :
  m_encoded_object(std::move(encoded_object)),
  m_value_descriptor(std::move(value_descriptor))
{}

RemoteDescriptorImpl2::RemoteDescriptorImpl2(std::unique_ptr<codable::protos::DescriptorObject> encoded_object) :
  m_serialized_object(std::move(encoded_object))
{}

codable::protos::DescriptorObject& RemoteDescriptorImpl2::encoded_object() const
{
    return *m_serialized_object;
}

memory::buffer RemoteDescriptorImpl2::to_bytes(std::shared_ptr<memory::memory_resource> mr) const
{
    // Allocate enough bytes to hold the encoded object
    auto buffer = memory::buffer(m_serialized_object->ByteSizeLong(), mr);

    this->to_bytes(buffer);

    return buffer;
}

memory::buffer_view RemoteDescriptorImpl2::to_bytes(memory::buffer_view buffer) const
{
    if (!m_serialized_object->SerializeToArray(buffer.data(), buffer.bytes()))
    {
        LOG(FATAL) << "Failed to serialize EncodedObjectProto to bytes";
    }

    return buffer;
}

std::shared_ptr<RemoteDescriptorImpl2> RemoteDescriptorImpl2::from_local(
    std::unique_ptr<LocalDescriptor2> local_desc,
    data_plane::DataPlaneResources2& data_plane_resources)
{
    auto remote_object = std::make_unique<codable::protos::DescriptorObject>();

    // Transfer the info object
    remote_object->set_instance_id(data_plane_resources.get_instance_id());
    remote_object->set_tokens(std::numeric_limits<uint64_t>::max());

    // Loop over all local payloads and convert them to remote payloads

    for (const auto& local_payload : local_desc->encoded_object().proto().payloads())
    {
        auto* remote_payload = remote_object->add_payloads();

        remote_payload->set_memory_kind(local_payload.memory_kind());

        // If payload is an EagerMessage, we simply add as payload
        if (local_payload.has_eager_msg())
        {
            auto* eager_msg = remote_payload->mutable_eager_msg();
            eager_msg->set_data(local_payload.eager_msg().data());
            continue;
        }

        auto* deferred_msg = remote_payload->mutable_deferred_msg();
        auto deferred_local_msg = local_payload.deferred_msg();

        auto ucx_block = data_plane_resources.registration_cache().lookup(deferred_local_msg.address());

        if (!ucx_block.has_value())
        {
            // Need to register the memory
            ucx_block = data_plane_resources.registration_cache().add_block(deferred_local_msg.address(),
                                                                            deferred_local_msg.bytes());
        }

        bool should_cache = false;  // Not sure what to set this

        deferred_msg->set_address(deferred_local_msg.address());
        deferred_msg->set_bytes(deferred_local_msg.bytes());
        deferred_msg->set_memory_block_address(reinterpret_cast<std::uint64_t>(ucx_block->data()));
        deferred_msg->set_memory_block_size(ucx_block->bytes());
        deferred_msg->set_remote_key(ucx_block->packed_remote_keys());
    }

    auto remote_descriptor = std::shared_ptr<RemoteDescriptorImpl2>(
        new RemoteDescriptorImpl2(std::move(remote_object)));
    // data_plane_resources.register_remote_decriptor(remote_descriptor);

    return remote_descriptor;
}

std::shared_ptr<RemoteDescriptorImpl2> RemoteDescriptorImpl2::from_bytes(memory::const_buffer_view view)
{
    auto encoded_obj_proto = std::make_unique<codable::protos::DescriptorObject>();

    if (!encoded_obj_proto->ParseFromArray(view.data(), view.bytes()))
    {
        LOG(FATAL) << "Failed to parse EncodedObjectProto from bytes";
    }

    return std::shared_ptr<RemoteDescriptorImpl2>(new RemoteDescriptorImpl2(std::move(encoded_obj_proto)));
}

RemoteDescriptor2::RemoteDescriptor2(std::unique_ptr<codable::protos::DescriptorObject> encoded_object) :
  m_impl(new RemoteDescriptorImpl2(std::move(encoded_object)))
{}

RemoteDescriptor2::RemoteDescriptor2(std::shared_ptr<RemoteDescriptorImpl2> impl) : m_impl(std::move(impl)) {}

codable::protos::DescriptorObject& RemoteDescriptor2::encoded_object() const
{
    return m_impl->encoded_object();
}

memory::buffer RemoteDescriptor2::to_bytes(std::shared_ptr<memory::memory_resource> mr) const
{
    return m_impl->to_bytes(mr);
}

memory::buffer_view RemoteDescriptor2::to_bytes(memory::buffer_view buffer) const
{
    return m_impl->to_bytes(buffer);
}

std::unique_ptr<RemoteDescriptor2> RemoteDescriptor2::from_local(std::unique_ptr<LocalDescriptor2> local_desc,
                                                                 data_plane::DataPlaneResources2& data_plane_resources)
{
    return std::unique_ptr<RemoteDescriptor2>(
        new RemoteDescriptor2(RemoteDescriptorImpl2::from_local(std::move(local_desc), data_plane_resources)));
}

std::unique_ptr<RemoteDescriptor2> RemoteDescriptor2::from_bytes(memory::const_buffer_view view)
{
    return std::unique_ptr<RemoteDescriptor2>(new RemoteDescriptor2(RemoteDescriptorImpl2::from_bytes(view)));
}

codable::protos::DescriptorObject& Descriptor2::encoded_object() const
{
    return m_encoded_object->proto();
}

std::shared_ptr<Descriptor2> Descriptor2::create(std::any value, data_plane::DataPlaneResources2& data_plane_resources)
{
    return std::shared_ptr<Descriptor2>(new Descriptor2(std::move(value), data_plane_resources));
}

std::shared_ptr<Descriptor2> Descriptor2::create(memory::buffer_view view, data_plane::DataPlaneResources2& data_plane_resources)
{
    auto descriptor = std::make_unique<codable::DescriptorObjectHandler>();

    if (!descriptor->proto().ParseFromArray(view.data(), view.bytes()))
    {
        LOG(FATAL) << "Failed to parse EncodedObjectProto from bytes";
    }

    auto mr = memory::malloc_memory_resource::instance();

    std::vector<std::shared_ptr<ucxx::Request>> requests;
    std::vector<memory::buffer> buffers;

    // Get the endpoint of the remote descriptor
    auto ep = data_plane_resources.find_endpoint(descriptor->proto().instance_id());

    // Loop over all remote payloads and convert them to local payloads
    for (const auto& remote_payload : descriptor->proto().payloads())
    {
        // If payload is an EagerMessage, we do not need to do any pulling
        if (remote_payload.has_eager_msg())
        {
            continue;
        }

        // Get the DeferredMessage of the remote payload
        auto deferred_remote_msg = remote_payload.deferred_msg();

        // Allocate the memory needed for this and prevent it from going out-of-scope before request completes
        buffers.emplace_back(deferred_remote_msg.bytes(), mr);

        // now issue the request
        requests.push_back(data_plane_resources.memory_recv_async(ep,
                                                                  buffers.back(),
                                                                  deferred_remote_msg.address(),
                                                                  deferred_remote_msg.remote_key().data()));

        deferred_remote_msg.set_address(reinterpret_cast<uintptr_t>(buffers.back().data()));
        deferred_remote_msg.set_bytes(buffers.back().bytes());
    }

    // Now, we need to wait for all requests to be complete
    data_plane_resources.wait_requests(requests);

    // For the remote descriptor message, send decrement to the remote resources
    remote_descriptor::RemoteDescriptorDecrementMessage dec_message;
    dec_message.object_id = descriptor->proto().object_id();
    dec_message.tokens    = descriptor->proto().tokens();

    // TODO(Peter): Define `ucxx::AmReceiverCallbackInfo` at central place, must be known by all MRC processes.
    // Send a decrement message using custom AM receiver callback
    auto decrement_request = ep->amSend(&dec_message,
                                        sizeof(remote_descriptor::RemoteDescriptorDecrementMessage),
                                        UCS_MEMORY_TYPE_HOST,
                                        ucxx::AmReceiverCallbackInfo("MRC", 0));

    return std::shared_ptr<Descriptor2>(new Descriptor2(std::move(descriptor), data_plane_resources));
}

void Descriptor2::setup_remote_payloads()
{
    auto& remote_object = m_encoded_object->proto();

    // Transfer the info object
    remote_object.set_instance_id(m_data_plane_resources.get_instance_id());
    remote_object.set_tokens(std::numeric_limits<uint64_t>::max());

    // Loop over all local payloads and convert them to remote payloads

    for (auto& payload : *(remote_object.mutable_payloads()))
    {
        // If payload is an EagerMessage, we simply continue
        if (payload.has_eager_msg())
        {
            continue;
        }

        auto* deferred_msg = payload.mutable_deferred_msg();

        auto ucx_block = m_data_plane_resources.registration_cache().lookup(deferred_msg->address());

        if (!ucx_block.has_value())
        {
            // Need to register the memory
            ucx_block = m_data_plane_resources.registration_cache().add_block(deferred_msg->address(),
                                                                            deferred_msg->bytes());
        }

        deferred_msg->set_memory_block_address(reinterpret_cast<std::uint64_t>(ucx_block->data()));
        deferred_msg->set_memory_block_size(ucx_block->bytes());
        deferred_msg->set_remote_key(ucx_block->packed_remote_keys());
    }

    m_data_plane_resources.register_remote_decriptor(shared_from_this());
}
}  // namespace mrc::runtime

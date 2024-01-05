/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/codable/encoded_object.hpp"

#include "mrc/codable/memory.hpp"
#include "mrc/memory/buffer.hpp"
#include "mrc/memory/resources/memory_resource.hpp"
#include "mrc/protos/codable.pb.h"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>

#include <memory>
#include <type_traits>

namespace mrc::codable {

EncodedStorage::EncodedStorage(std::unique_ptr<mrc::codable::IDecodableStorage> encoding) :
  m_encoding(std::move(encoding))
{
    CHECK(m_encoding);
}

mrc::codable::protos::LocalSerializedObject& LocalSerializedWrapper::proto()
{
    return m_proto;
}

const mrc::codable::protos::LocalSerializedObject& LocalSerializedWrapper::proto() const
{
    return m_proto;
}

bool LocalSerializedWrapper::operator==(const LocalSerializedWrapper& other) const
{
    return m_proto.SerializeAsString() == other.m_proto.SerializeAsString();
}

bool LocalSerializedWrapper::has_current_object() const
{
    return !m_object_idx_stack.empty();
}

void LocalSerializedWrapper::reset_current_object_idx()
{
    CHECK(m_object_idx_stack.empty()) << "Cannot reset while currently in use";

    m_object_counter = 0;
}

size_t LocalSerializedWrapper::get_current_object_idx() const
{
    if (!this->has_current_object())
    {
        throw std::runtime_error("Current object index is not set");
    }

    return m_object_idx_stack.top();
}

size_t LocalSerializedWrapper::push_current_object_idx(std::type_index type_info)
{
    auto created_obj_idx = this->add_object(type_info);

    return created_obj_idx;
}

size_t LocalSerializedWrapper::push_current_object_idx(std::type_index type_info) const
{
    // There is already an object at this index. Check the type index to ensure its the same
    const auto& obj = this->get_object(m_object_counter);

    if (obj.type_index_hash() != type_info.hash_code())
    {
        throw std::runtime_error("Type index mismatch");
    }

    m_object_idx_stack.push(m_object_counter++);

    return this->get_current_object_idx();
}

void LocalSerializedWrapper::pop_current_object_idx(size_t pushed_object_idx)
{
    if (this->get_current_object_idx() != pushed_object_idx)
    {
        throw std::runtime_error("Object index mismatch. Must pop objects with the same value returned from push");
    }

    m_object_idx_stack.pop();
}

void LocalSerializedWrapper::pop_current_object_idx(size_t pushed_object_idx) const
{
    if (this->get_current_object_idx() != pushed_object_idx)
    {
        throw std::runtime_error("Object index mismatch. Must pop objects with the same value returned from push");
    }

    m_object_idx_stack.pop();
}

size_t LocalSerializedWrapper::objects_size() const
{
    return m_proto.info().objects_size();
}

size_t LocalSerializedWrapper::descriptors_size() const
{
    return m_proto.info().descriptors_size();
}

size_t LocalSerializedWrapper::payloads_size() const
{
    return m_proto.payloads_size();
}

protos::Object& LocalSerializedWrapper::get_object(size_t idx)
{
    return *m_proto.mutable_info()->mutable_objects(idx);
}

const protos::Object& LocalSerializedWrapper::get_object(size_t idx) const
{
    return m_proto.info().objects(idx);
}

size_t LocalSerializedWrapper::get_descriptor_idx(size_t object_idx, size_t desc_offset) const
{
    // Get the object
    const auto& obj = this->get_object(object_idx);

    return obj.descriptor_idxs(desc_offset);
}

protos::MemoryDescriptor& LocalSerializedWrapper::get_descriptor(size_t desc_idx)
{
    return *m_proto.mutable_info()->mutable_descriptors(desc_idx);
}

const protos::MemoryDescriptor& LocalSerializedWrapper::get_descriptor(size_t desc_idx) const
{
    return m_proto.info().descriptors(desc_idx);
}

protos::MemoryDescriptor& LocalSerializedWrapper::get_descriptor_from_offset(size_t desc_offset,
                                                                             std::optional<size_t> object_idx)
{
    if (!object_idx.has_value())
    {
        // This will throw an error if not set
        object_idx = this->get_current_object_idx();
    }

    return this->get_descriptor(this->get_descriptor_idx(object_idx.value(), desc_offset));
}

const protos::MemoryDescriptor& LocalSerializedWrapper::get_descriptor_from_offset(
    size_t desc_offset,
    std::optional<size_t> object_idx) const
{
    if (!object_idx.has_value())
    {
        // This will throw an error if not set
        object_idx = this->get_current_object_idx();
    }

    return this->get_descriptor(this->get_descriptor_idx(object_idx.value(), desc_offset));
}

protos::LocalPayload& LocalSerializedWrapper::get_payload(size_t idx)
{
    return *m_proto.mutable_payloads(idx);
}

const protos::LocalPayload& LocalSerializedWrapper::get_payload(size_t idx) const
{
    return m_proto.payloads(idx);
}

// protos::Object& LocalSerializedWrapper::add_object()
// {
//     auto* obj = m_proto.mutable_info()->add_objects();

//     // Set the parent object index to the current object index
//     obj->set_parent_object_idx(m_current_obj_idx.value_or(-1));

//     return *obj;
// }

size_t LocalSerializedWrapper::add_object(std::type_index type_info)
{
    auto* obj = m_proto.mutable_info()->add_objects();

    // Set the parent object index to the current object index
    obj->set_parent_object_idx(this->has_current_object() ? this->get_current_object_idx() : -1);
    obj->set_type_index_hash(type_info.hash_code());

    // Set thew new current object index
    m_object_idx_stack.push(m_object_counter++);

#ifdef DEBUG
    obj->set_debug_info(type_name(type_info));
#endif

    return this->get_current_object_idx();
}

protos::MemoryDescriptor& LocalSerializedWrapper::add_descriptor()
{
    // Get the current object
    auto& obj = this->get_object(this->get_current_object_idx());

    // Create a new descriptor
    auto* descriptor = m_proto.mutable_info()->add_descriptors();

    // Add the list of descriptor idxs to the object
    obj.add_descriptor_idxs(this->descriptors_size() - 1);

    return *descriptor;
}

size_t LocalSerializedWrapper::add_descriptor(memory::const_buffer_view view, DescriptorKind kind)
{
    throw std::runtime_error("Not implemented");
}

// bool LocalSerializedWrapper::context_acquired() const
// {
//     std::lock_guard lock(m_mutex);
//     return m_context_acquired;
// }

// obj_idx_t LocalSerializedWrapper::push_context(std::type_index type_index)
// {
//     std::lock_guard lock(m_mutex);
//     m_context_acquired = true;

//     auto initial_parent_object_idx = m_parent.value_or(-1);

//     // Set the parent idx
//     m_parent = this->objects_size();

//     auto* obj = m_proto.mutable_info()->add_objects();

//     obj->set_type_index_hash(type_index.hash_code());
//     obj->set_starting_descriptor_idx(this->descriptors_size());
//     obj->set_parent_object_idx(initial_parent_object_idx);

//     return m_parent.value();
// }

// void LocalSerializedWrapper::pop_context(obj_idx_t object_idx)
// {
//     std::lock_guard lock(m_mutex);
//     m_parent = object_idx;
//     if (object_idx == -1)
//     {
//         m_context_acquired = false;
//     }
// }

// obj_idx_t LocalSerializedWrapper::push_decode_context(std::type_index type_index) const
// {
//     std::lock_guard lock(m_mutex);
//     // m_context_acquired = true;

//     auto initial_parent_object_idx = m_parent.value_or(-1);

//     const auto& obj = m_proto.info().objects().at(initial_parent_object_idx);

//     // Check that the type index matches
//     CHECK_EQ(obj.type_index_hash(), type_index.hash_code());

//     // m_parent = this->objects_size();

//     return initial_parent_object_idx;
// }
// void LocalSerializedWrapper::pop_decode_context(obj_idx_t object_idx) const
// {
//     std::lock_guard lock(m_mutex);
//     // m_parent = object_idx;
//     if (object_idx == -1)
//     {
//         // m_context_acquired = false;
//     }
// }

idx_t LocalSerializedWrapper::add_eager_descriptor(memory::const_buffer_view view)
{
    const void* data = view.data();
    size_t bytes     = view.bytes();

    if (view.kind() == memory::memory_kind::device)
    {
        LOG(FATAL) << "Device memory is not supported yet";

        // TODO(MDD): Copy data from device to host and update data
    }

    auto* descriptor = m_proto.mutable_info()->add_descriptors();

    auto* eager_descriptor = descriptor->mutable_eager_desc();

    eager_descriptor->set_data(data, bytes);

    return this->descriptors_size();
}

idx_t LocalSerializedWrapper::add_remote_memory_descriptor(uint64_t instance_id,
                                                           uintptr_t address,
                                                           size_t bytes,
                                                           uintptr_t memory_block_address,
                                                           size_t memory_block_size,
                                                           void* remote_key,
                                                           memory::memory_kind memory_kind)
{
    auto* descriptor = m_proto.mutable_info()->add_descriptors();

    // auto* remote_memory_descriptor = descriptor->mutable_remote_desc();

    // remote_memory_descriptor->set_instance_id(instance_id);
    // remote_memory_descriptor->set_address(address);
    // remote_memory_descriptor->set_bytes(bytes);
    // remote_memory_descriptor->set_memory_block_address(memory_block_address);
    // remote_memory_descriptor->set_memory_block_size(memory_block_size);
    // remote_memory_descriptor->set_memory_kind(mrc::codable::encode_memory_type(memory_kind));
    // // remote_memory_descriptor->set_remote_key(remote_key);

    return this->descriptors_size();
}

// std::size_t LocalSerializedWrapper::type_index_hash_for_object(const obj_idx_t& object_idx) const
// {
//     CHECK_LT(object_idx, m_proto.info().objects_size());
//     return m_proto.info().objects().at(object_idx).type_index_hash();
// }

// idx_t LocalSerializedWrapper::start_idx_for_object(const obj_idx_t& object_idx) const
// {
//     CHECK_LT(object_idx, m_proto.info().objects_size());
//     return m_proto.info().objects().at(object_idx).starting_descriptor_idx();
// }

// std::optional<obj_idx_t> LocalSerializedWrapper::parent_obj_idx_for_object(const obj_idx_t& object_idx) const
// {
//     CHECK_LT(object_idx, m_proto.info().objects_size());
//     auto parent_object_idx = m_proto.info().objects().at(object_idx).parent_object_idx();
//     if (parent_object_idx < 0)
//     {
//         return std::nullopt;
//     }
//     return parent_object_idx;
// }

mrc::codable::protos::SerializedInfo& LocalSerializedWrapper::mutable_info()
{
    return *m_proto.mutable_info();
}

const ::google::protobuf::RepeatedPtrField<::mrc::codable::protos::LocalPayload>& LocalSerializedWrapper::payloads()
    const
{
    return m_proto.payloads();
}

memory::buffer LocalSerializedWrapper::to_bytes(std::shared_ptr<memory::memory_resource> mr) const
{
    // Allocate enough bytes to hold the encoded object
    auto buffer = memory::buffer(m_proto.ByteSizeLong(), mr);

    this->to_bytes(buffer);

    return buffer;
}

memory::buffer_view LocalSerializedWrapper::to_bytes(memory::buffer_view buffer) const
{
    if (!m_proto.SerializeToArray(buffer.data(), buffer.bytes()))
    {
        LOG(FATAL) << "Failed to serialize EncodedObjectProto to bytes";
    }

    return buffer;
}

std::unique_ptr<LocalSerializedWrapper> LocalSerializedWrapper::from_bytes(memory::const_buffer_view view)
{
    auto encoded_obj_proto = std::make_unique<LocalSerializedWrapper>();

    if (!encoded_obj_proto->m_proto.ParseFromArray(view.data(), view.bytes()))
    {
        LOG(FATAL) << "Failed to parse EncodedObjectProto from bytes";
    }

    return encoded_obj_proto;
}

}  // namespace mrc::codable

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

#include <glog/logging.h>

#include <memory>

namespace mrc::codable {

EncodedStorage::EncodedStorage(std::unique_ptr<mrc::codable::IDecodableStorage> encoding) :
  m_encoding(std::move(encoding))
{
    CHECK(m_encoding);
}

bool EncodedObjectProto::operator==(const EncodedObjectProto& other) const
{
    return m_proto.SerializeAsString() == other.m_proto.SerializeAsString();
}

size_t EncodedObjectProto::objects_size() const
{
    return m_proto.objects_size();
}

size_t EncodedObjectProto::descriptors_size() const
{
    return m_proto.descriptors_size();
}

bool EncodedObjectProto::context_acquired() const
{
    std::lock_guard lock(m_mutex);
    return m_context_acquired;
}

obj_idx_t EncodedObjectProto::push_context(std::type_index type_index)
{
    std::lock_guard lock(m_mutex);
    m_context_acquired = true;

    auto initial_parent_object_idx = m_parent.value_or(-1);

    auto* obj = m_proto.add_objects();

    obj->set_type_index_hash(type_index.hash_code());
    obj->set_starting_descriptor_idx(this->descriptors_size());
    obj->set_parent_object_idx(initial_parent_object_idx);

    m_parent = this->objects_size();

    return initial_parent_object_idx;
}

void EncodedObjectProto::pop_context(obj_idx_t object_idx)
{
    std::lock_guard lock(m_mutex);
    m_parent = object_idx;
    if (object_idx == -1)
    {
        m_context_acquired = false;
    }
}

idx_t EncodedObjectProto::add_eager_descriptor(memory::const_buffer_view view)
{
    const void* data = view.data();
    size_t bytes     = view.bytes();

    if (view.kind() == memory::memory_kind::device)
    {
        LOG(FATAL) << "Device memory is not supported yet";

        // TODO(MDD): Copy data from device to host and update data
    }

    auto* descriptor = m_proto.add_descriptors();

    auto* eager_descriptor = descriptor->mutable_eager_desc();

    eager_descriptor->set_data(data, bytes);

    return this->descriptors_size();
}

idx_t EncodedObjectProto::add_remote_memory_descriptor(uint64_t instance_id,
                                                       uintptr_t address,
                                                       size_t bytes,
                                                       uintptr_t memory_block_address,
                                                       size_t memory_block_size,
                                                       void* remote_key,
                                                       memory::memory_kind memory_kind)
{
    auto* descriptor = m_proto.add_descriptors();

    auto* remote_memory_descriptor = descriptor->mutable_remote_desc();

    remote_memory_descriptor->set_instance_id(instance_id);
    remote_memory_descriptor->set_address(address);
    remote_memory_descriptor->set_bytes(bytes);
    remote_memory_descriptor->set_memory_block_address(memory_block_address);
    remote_memory_descriptor->set_memory_block_size(memory_block_size);
    remote_memory_descriptor->set_memory_kind(mrc::codable::encode_memory_type(memory_kind));
    // remote_memory_descriptor->set_remote_key(remote_key);

    return this->descriptors_size();
}

memory::buffer EncodedObjectProto::to_bytes(std::shared_ptr<memory::memory_resource> mr) const
{
    // Allocate enough bytes to hold the encoded object
    auto buffer = memory::buffer(m_proto.ByteSizeLong(), mr);

    this->to_bytes(buffer);

    return buffer;
}

memory::buffer_view EncodedObjectProto::to_bytes(memory::buffer_view buffer) const
{
    if (!m_proto.SerializeToArray(buffer.data(), buffer.bytes()))
    {
        LOG(FATAL) << "Failed to serialize EncodedObjectProto to bytes";
    }

    return buffer;
}

std::unique_ptr<EncodedObjectProto> EncodedObjectProto::from_bytes(memory::const_buffer_view view)
{
    auto encoded_obj_proto = std::make_unique<EncodedObjectProto>();

    if (!encoded_obj_proto->m_proto.ParseFromArray(view.data(), view.bytes()))
    {
        LOG(FATAL) << "Failed to parse EncodedObjectProto from bytes";
    }

    return encoded_obj_proto;
}

}  // namespace mrc::codable

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

#include <srf/codable/encoded_object.hpp>

#include <srf/protos/codable.pb.h>
#include <srf/codable/memory_resources.hpp>
#include <srf/memory/block.hpp>
#include <srf/memory/memory_kind.hpp>
#include <srf/utils/thread_local_shared_pointer.hpp>

#include <google/protobuf/any.pb.h>
#include <google/protobuf/message.h>

#include <cstdint>  // for uint64_t
#include <memory>   // for __shared_ptr_access, shared_ptr
#include <ostream>  // for operator<<

namespace srf::codable {

static memory::memory_kind decode_memory_type(const protos::MemoryKind& proto_kind)
{
    switch (proto_kind)
    {
    case protos::MemoryKind::Host:
        return memory::memory_kind::host;
    case protos::MemoryKind::Pinned:
        return memory::memory_kind::pinned;
    case protos::MemoryKind::Device:
        return memory::memory_kind::device;
    case protos::MemoryKind::Managed:
        return memory::memory_kind::managed;
    default:
        LOG(FATAL) << "unhandled protos::MemoryKind";
    };

    return memory::memory_kind::none;
}

static protos::MemoryKind encode_memory_type(memory::memory_kind mem_kind)
{
    switch (mem_kind)
    {
    case memory::memory_kind::host:
        return protos::MemoryKind::Host;
    case memory::memory_kind::pinned:
        return protos::MemoryKind::Pinned;
    case memory::memory_kind::device:
        return protos::MemoryKind::Device;
    case memory::memory_kind::managed:
        return protos::MemoryKind::Managed;
    default:
        LOG(FATAL) << "unhandled protos::MemoryKind";
    };

    return protos::MemoryKind::None;
}

memory::block EncodedObject::decode_descriptor(const protos::RemoteDescriptor& desc)
{
    return memory::block(
        reinterpret_cast<void*>(desc.remote_address()), desc.remote_bytes(), decode_memory_type(desc.memory_kind()));
}

protos::RemoteDescriptor EncodedObject::encode_descriptor(memory::const_block view)
{
    protos::RemoteDescriptor desc;
    desc.set_remote_address(reinterpret_cast<std::uint64_t>(view.data()));
    desc.set_remote_bytes(view.bytes());
    desc.set_memory_kind(encode_memory_type(view.kind()));

    return desc;
}

const protos::EncodedObject& EncodedObject::proto() const
{
    return m_proto;
}

memory::const_block EncodedObject::memory_block(std::size_t idx) const
{
    DCHECK_LT(idx, descriptor_count());
    CHECK(m_proto.descriptors().at(idx).has_remote_desc());
    return decode_descriptor(m_proto.descriptors().at(idx).remote_desc());
}

const protos::EagerDescriptor& EncodedObject::eager_descriptor(std::size_t idx) const
{
    DCHECK_LT(idx, descriptor_count());
    CHECK(m_proto.descriptors().at(idx).has_eager_desc());
    return m_proto.descriptors().at(idx).eager_desc();
}

memory::block EncodedObject::mutable_memory_block(std::size_t idx) const
{
    CHECK(m_context_acquired);
    DCHECK_LT(idx, descriptor_count());
    CHECK(m_proto.descriptors().at(idx).has_remote_desc());
    return decode_descriptor(m_proto.descriptors().at(idx).remote_desc());
}

std::size_t EncodedObject::descriptor_count() const
{
    return m_proto.descriptors_size();
}

std::size_t EncodedObject::object_count() const
{
    return m_proto.objects_size();
}

std::size_t EncodedObject::type_index_hash_for_object(std::size_t idx) const
{
    DCHECK_LT(idx, object_count());
    return m_proto.objects().at(idx).type_index_hash();
}

std::size_t EncodedObject::start_idx_for_object(std::size_t idx) const
{
    DCHECK_LT(idx, object_count());
    return m_proto.objects().at(idx).desc_id();
}

std::size_t EncodedObject::add_meta_data(const google::protobuf::Message& meta_data)
{
    CHECK(m_context_acquired);
    auto index = m_proto.descriptors_size();
    auto* desc = m_proto.add_descriptors();
    desc->mutable_meta_data_desc()->mutable_meta_data()->PackFrom(meta_data);
    return index;
}

std::size_t EncodedObject::add_memory_block(memory::const_block view)
{
    CHECK(m_context_acquired);
    auto count = descriptor_count();
    auto* desc = m_proto.add_descriptors()->mutable_remote_desc();
    *desc      = encode_descriptor(view);
    return count;
}

std::size_t EncodedObject::add_host_buffer(std::size_t bytes)
{
    CHECK(m_context_acquired);
    LOG(FATAL) << "disabled path - awaiting runtime resources";
    // auto view = utils::ThreadLocalSharedPointer<codable::MemoryResources>::get()->host_resource_view();
    // return add_buffer(view, bytes);
}

std::size_t EncodedObject::add_device_buffer(std::size_t bytes)
{
    CHECK(m_context_acquired);
    LOG(FATAL) << "disabled path - awaiting runtime resources";
    // auto view = utils::ThreadLocalSharedPointer<codable::MemoryResources>::get()->device_resource_view();
    // return add_buffer(view, bytes);
}

std::size_t EncodedObject::add_eager_buffer(const void* data, std::size_t bytes)
{
    CHECK(m_context_acquired);
    auto count                    = descriptor_count();
    protos::EagerDescriptor* desc = m_proto.add_descriptors()->mutable_eager_desc();
    desc->set_data(data, bytes);
    return count;
}

EncodedObject::ContextGuard::ContextGuard(EncodedObject& encoded_object, std::type_index type_index) :
  m_encoded_object(encoded_object)
{
    CHECK(m_encoded_object.m_context_acquired == false);
    m_encoded_object.m_context_acquired = true;
    m_encoded_object.add_type_index(type_index);
}

EncodedObject::ContextGuard::~ContextGuard()
{
    CHECK(m_encoded_object.m_context_acquired);
    m_encoded_object.m_context_acquired = false;
}

void EncodedObject::add_type_index(std::type_index type_index)
{
    CHECK(m_context_acquired);
    auto* obj = m_proto.add_objects();
    obj->set_type_index_hash(type_index.hash_code());
    obj->set_desc_id(descriptor_count());
}

}  // namespace srf::codable

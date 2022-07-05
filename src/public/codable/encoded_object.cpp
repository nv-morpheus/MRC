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

#include "srf/codable/encoded_object.hpp"

#include "internal/data_plane/resources.hpp"
#include "internal/network/resources.hpp"
#include "internal/resources/forward.hpp"
#include "internal/resources/manager.hpp"

#include "srf/memory/buffer_view.hpp"
#include "srf/memory/memory_kind.hpp"
#include "srf/protos/codable.pb.h"

#include <google/protobuf/any.pb.h>
#include <google/protobuf/message.h>

#include <cstdint>
#include <ostream>

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

memory::buffer_view EncodedObject::decode_descriptor(const protos::RemoteMemoryDescriptor& desc)
{
    return memory::buffer_view(
        reinterpret_cast<void*>(desc.remote_address()), desc.remote_bytes(), decode_memory_type(desc.memory_kind()));
}

protos::RemoteMemoryDescriptor EncodedObject::encode_descriptor(memory::const_buffer_view view, std::string keys)
{
    protos::RemoteMemoryDescriptor desc;

    desc.set_remote_address(reinterpret_cast<std::uint64_t>(view.data()));
    desc.set_remote_bytes(view.bytes());
    desc.set_memory_kind(encode_memory_type(view.kind()));
    desc.set_remote_key(std::move(keys));
    return desc;
}

const protos::EncodedObject& EncodedObject::proto() const
{
    return m_proto;
}

memory::const_buffer_view EncodedObject::memory_block(std::size_t idx) const
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

memory::buffer_view EncodedObject::mutable_memory_block(std::size_t idx) const
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

std::size_t EncodedObject::add_memory_block(memory::const_buffer_view view)
{
    auto& resources = internal::resources::Manager::get_partition();
    CHECK(resources.network());
    auto ucx_block = resources.network()->data_plane().registration_cache().lookup(view.data());

    if (ucx_block)
    {
        CHECK(m_context_acquired);
        auto count = descriptor_count();
        auto* desc = m_proto.add_descriptors()->mutable_remote_desc();
        *desc      = encode_descriptor(view, ucx_block->packed_remote_keys());
        return count;
    }

    if (view.kind() == srf::memory::memory_kind::host)
    {
        DVLOG(10) << "unregistered host memory buffer view detected - a copy will be made";
        auto count = add_host_buffer(view.bytes());
        auto block = mutable_memory_block(count);
        std::memcpy(block.data(), view.data(), view.bytes());
        return count;
    }

    LOG(FATAL) << "unregistered device buffer is not supported at this time";
}

std::size_t EncodedObject::add_host_buffer(std::size_t bytes)
{
    CHECK(m_context_acquired);
    auto& resources = internal::resources::Manager::get_partition();
    return add_buffer(resources.host().make_buffer(bytes));
}

std::size_t EncodedObject::add_device_buffer(std::size_t bytes)
{
    CHECK(m_context_acquired);
    auto& resources = internal::resources::Manager::get_partition();
    CHECK(resources.device());
    return add_buffer(resources.device()->make_buffer(bytes));
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

std::size_t EncodedObject::add_buffer(memory::buffer&& buffer)
{
    CHECK(m_context_acquired);
    auto index       = add_memory_block(buffer);
    m_buffers[index] = std::move(buffer);
    return index;
}

}  // namespace srf::codable

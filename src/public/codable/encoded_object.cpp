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

#include "srf/memory/buffer_view.hpp"
#include "srf/memory/memory_kind.hpp"
#include "srf/protos/codable.pb.h"

#include <google/protobuf/any.pb.h>
#include <google/protobuf/message.h>

#include <cstdint>
#include <ostream>

namespace srf::codable {

memory::memory_kind decode_memory_type(const protos::MemoryKind& proto_kind)
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

protos::MemoryKind encode_memory_type(memory::memory_kind mem_kind)
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
        LOG(FATAL) << "unhandled srf::memory::memory_kind";
    };

    return protos::MemoryKind::None;
}

// EncodedObject

std::size_t EncodedObject::descriptor_count() const
{
    return m_proto.descriptors_size();
}

std::size_t EncodedObject::object_count() const
{
    return m_proto.objects_size();
}

std::size_t EncodedObject::type_index_hash_for_object(const obj_idx_t& object_idx) const
{
    DCHECK_LT(object_idx, object_count());
    return m_proto.objects().at(object_idx).type_index_hash();
}

idx_t EncodedObject::start_idx_for_object(const obj_idx_t& object_idx) const
{
    DCHECK_LT(object_idx, object_count());
    return m_proto.objects().at(object_idx).desc_id();
}

EncodedObject::EncodedObject(protos::EncodedObject proto) : m_proto(std::move(proto)) {}

const protos::EncodedObject& EncodedObject::proto() const
{
    return m_proto;
}

protos::EncodedObject& EncodedObject::proto()
{
    return m_proto;
}

const bool& EncodedObject::context_acquired() const
{
    return m_context_acquired;
}

std::size_t EncodedObject::add_meta_data(const google::protobuf::Message& meta_data)
{
    CHECK(m_context_acquired);
    auto index = m_proto.descriptors_size();
    auto* desc = m_proto.add_descriptors();
    desc->mutable_meta_data_desc()->mutable_meta_data()->PackFrom(meta_data);
    return index;
}

idx_t EncodedObject::copy_to_eager_descriptor(memory::const_buffer_view view)
{
    CHECK(m_context_acquired);
    auto count                    = descriptor_count();
    protos::EagerDescriptor* desc = m_proto.add_descriptors()->mutable_eager_desc();
    desc->set_data(view.data(), view.bytes());
    return count;
}

void EncodedObject::add_type_index(std::type_index type_index)
{
    CHECK(m_context_acquired);
    auto* obj = m_proto.add_objects();
    obj->set_type_index_hash(type_index.hash_code());
    obj->set_desc_id(descriptor_count());
}

// EncodedObject::ContextGuard

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

std::size_t EncodedObject::buffer_size(const idx_t& idx) const
{
    DCHECK_LT(idx, descriptor_count());
    const auto& desc = proto().descriptors().at(idx);

    CHECK(desc.has_eager_desc() || desc.has_remote_desc());

    if (desc.has_eager_desc())
    {
        return desc.eager_desc().data().size();
    }

    if (desc.has_remote_desc())
    {
        return desc.remote_desc().bytes();
    }

    return 0;
}
}  // namespace srf::codable

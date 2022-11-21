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

#include "internal/codable/codable_storage.hpp"

#include "internal/data_plane/resources.hpp"
#include "internal/network/resources.hpp"
#include "internal/resources/forward.hpp"
#include "internal/resources/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/ucx/memory_block.hpp"
#include "internal/ucx/remote_registration_cache.hpp"

#include "srf/codable/memory.hpp"
#include "srf/cuda/common.hpp"
#include "srf/memory/buffer_view.hpp"
#include "srf/memory/literals.hpp"
#include "srf/memory/memory_kind.hpp"
#include "srf/protos/codable.pb.h"
#include "srf/types.hpp"

#include <google/protobuf/any.pb.h>
#include <google/protobuf/message.h>

#include <cstdint>
#include <optional>
#include <ostream>

using namespace srf::memory::literals;

namespace srf::internal::codable {

CodableStorage::CodableStorage(resources::PartitionResources& resources) : m_resources(resources) {}
CodableStorage::CodableStorage(srf::codable::protos::EncodedObject proto, resources::PartitionResources& resources) :
  m_proto(std::move(proto)),
  m_resources(resources)
{}

srf::codable::IDecodableStorage& CodableStorage::decodable()
{
    return *this;
}

srf::codable::IEncodableStorage& CodableStorage::encodable()
{
    return *this;
}

std::optional<CodableStorage::idx_t> CodableStorage::register_memory_view(srf::memory::const_buffer_view view,
                                                                          bool force_register)
{
    CHECK(m_resources.network());
    bool should_cache = true;
    auto ucx_block    = m_resources.network()->data_plane().registration_cache().lookup(view.data());

    if (!ucx_block && !force_register && view.bytes() < 64_KiB)
    {
        return std::nullopt;
    }

    if (!ucx_block)
    {
        // register memory
        should_cache = false;
        m_resources.network()->ucx().registration_cache().add_block(view.data(), view.bytes());
        ucx_block = m_resources.network()->data_plane().registration_cache().lookup(view.data());
        m_temporary_registrations.push_back(std::move(view));
    }

    auto count = descriptor_count();
    auto* desc = mutable_proto().add_descriptors()->mutable_remote_desc();
    encode_descriptor(m_resources.network()->instance_id(), *desc, view, *ucx_block, should_cache);
    return count;
}

void CodableStorage::copy_to_buffer(idx_t buffer_idx, srf::memory::const_buffer_view view)
{
    auto search = m_buffers.find(buffer_idx);
    CHECK(search != m_buffers.end()) << "buffer_idx=" << buffer_idx << " was not created with create_buffer";

    auto& dst = search->second;
    CHECK_LE(dst.bytes(), view.bytes());

    // todo(ryan) - enumerate this to use explicit copy methods, e.g. std::memcpy, cudaMemcpy, cudaMemcpyAsync with
    // directional H2D/D2H. the resources object should provide a cuda stream pool per partition and a single runtime
    // stream on the off chance the partition doesn't have a device, but is exposed to device memory.
    SRF_CHECK_CUDA(cudaMemcpy(dst.data(), view.data(), view.bytes(), cudaMemcpyDefault));
}

CodableStorage::idx_t CodableStorage::copy_to_eager_descriptor(srf::memory::const_buffer_view view)
{
    CHECK(context_acquired());
    auto count = descriptor_count();
    auto* desc = mutable_proto().add_descriptors()->mutable_eager_desc();
    desc->set_data(view.data(), view.bytes());
    return count;
}

CodableStorage::idx_t CodableStorage::create_memory_buffer(std::uint64_t bytes)
{
    CHECK(context_acquired());
    auto buffer = m_resources.host().make_buffer(bytes);
    auto idx    = register_memory_view(buffer);
    CHECK(idx);
    m_buffers[*idx] = std::move(buffer);
    return *idx;
}

void CodableStorage::encode_descriptor(const InstanceID& instance_id,
                                       srf::codable::protos::RemoteMemoryDescriptor& desc,
                                       srf::memory::const_buffer_view view,
                                       const ucx::MemoryBlock& ucx_block,
                                       bool should_cache)
{
    desc.set_instance_id(instance_id);
    desc.set_address(reinterpret_cast<std::uint64_t>(view.data()));
    desc.set_bytes(view.bytes());
    desc.set_memory_block_address(reinterpret_cast<std::uint64_t>(ucx_block.data()));
    desc.set_memory_block_size(ucx_block.bytes());
    desc.set_memory_kind(srf::codable::encode_memory_type(view.kind()));
    desc.set_remote_key(ucx_block.packed_remote_keys());
    desc.set_should_cache(should_cache);
}

srf::memory::buffer_view CodableStorage::decode_descriptor(const srf::codable::protos::RemoteMemoryDescriptor& desc)
{
    return {
        reinterpret_cast<void*>(desc.address()), desc.bytes(), srf::codable::decode_memory_type(desc.memory_kind())};
}

srf::codable::protos::EncodedObject& CodableStorage::get_mutable_proto()
{
    return m_proto;
}

const srf::codable::protos::EncodedObject& CodableStorage::get_proto() const
{
    return m_proto;
}

resources::PartitionResources& CodableStorage::resources() const
{
    return m_resources;
}

CodableStorage::obj_idx_t CodableStorage::push_context(std::type_index type_index)
{
    std::lock_guard lock(m_mutex);
    m_context_acquired = true;

    auto initial_parent_object_idx = m_parent.value_or(-1);

    auto* obj = m_proto.add_objects();
    obj->set_type_index_hash(type_index.hash_code());
    obj->set_starting_descriptor_idx(descriptor_count());
    obj->set_parent_object_idx(initial_parent_object_idx);

    m_parent = object_count();

    return initial_parent_object_idx;
}

void CodableStorage::pop_context(obj_idx_t object_idx)
{
    std::lock_guard lock(m_mutex);
    m_parent = object_idx;
    if (object_idx == -1)
    {
        m_context_acquired = false;
    }
}

bool CodableStorage::context_acquired() const
{
    std::lock_guard lock(m_mutex);
    return m_context_acquired;
}

srf::codable::protos::EncodedObject& CodableStorage::mutable_proto()
{
    return get_mutable_proto();
}

srf::memory::buffer_view CodableStorage::mutable_host_buffer_view(const idx_t& buffer_idx)
{
    CHECK_LT(buffer_idx, descriptor_count());
    const auto& desc = mutable_proto().descriptors().at(buffer_idx);

    CHECK(desc.has_remote_desc());
    const auto& rd = desc.remote_desc();
    CHECK(rd.memory_kind() == srf::codable::protos::MemoryKind::Host ||
          rd.memory_kind() == srf::codable::protos::MemoryKind::Pinned);

    return {reinterpret_cast<void*>(rd.address()), rd.bytes(), srf::codable::decode_memory_type(rd.memory_kind())};
}

CodableStorage::idx_t CodableStorage::add_meta_data(const google::protobuf::Message& meta_data)
{
    CHECK(context_acquired());
    auto index = m_proto.descriptors_size();
    auto* desc = m_proto.add_descriptors();
    CHECK(desc->mutable_meta_data_desc()->mutable_meta_data()->PackFrom(meta_data));
    return index;
}

}  // namespace srf::internal::codable
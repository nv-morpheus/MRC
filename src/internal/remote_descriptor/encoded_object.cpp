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

#include "internal/remote_descriptor/encoded_object.hpp"

#include "internal/data_plane/resources.hpp"
#include "internal/network/resources.hpp"
#include "internal/resources/forward.hpp"
#include "internal/resources/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/ucx/memory_block.hpp"

#include "srf/codable/memory.hpp"
#include "srf/memory/buffer_view.hpp"
#include "srf/memory/literals.hpp"
#include "srf/memory/memory_kind.hpp"
#include "srf/memory/resources/host/malloc_memory_resource.hpp"
#include "srf/protos/codable.pb.h"
#include "srf/types.hpp"

#include <google/protobuf/any.pb.h>
#include <google/protobuf/message.h>

#include <cstdint>
#include <ostream>

using namespace srf::memory::literals;

namespace srf::internal::remote_descriptor {

std::optional<codable::idx_t> EncodedObject::register_memory_view(srf::memory::const_buffer_view view,
                                                                  bool force_register)
{
    CHECK(context_acquired());
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
    auto* desc = proto().add_descriptors()->mutable_remote_desc();
    encode_descriptor(m_resources.network()->instance_id(), *desc, view, *ucx_block, should_cache);
    return count;
}

codable::idx_t EncodedObject::copy_to_eager_descriptor(srf::memory::const_buffer_view view)
{
    CHECK(context_acquired());
    auto count = descriptor_count();
    auto* desc = proto().add_descriptors()->mutable_eager_desc();
    desc->set_data(view.data(), view.bytes());
    return count;
}

codable::idx_t EncodedObject::create_memory_buffer(std::uint64_t bytes)
{
    CHECK(context_acquired());
    auto buffer = m_resources.host().make_buffer(bytes);
    auto idx    = register_memory_view(buffer);
    CHECK(idx);
    m_buffers[*idx] = std::move(buffer);
    return *idx;
}

srf::memory::buffer_view EncodedObject::mutable_memory_buffer(const codable::idx_t& idx) const
{
    CHECK(context_acquired());
    auto search = m_buffers.find(idx);
    CHECK(search != m_buffers.end());
    return search->second;
}

void EncodedObject::encode_descriptor(const InstanceID& instance_id,
                                      codable::protos::RemoteMemoryDescriptor& desc,
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

srf::memory::buffer_view EncodedObject::decode_descriptor(const codable::protos::RemoteMemoryDescriptor& desc)
{
    return srf::memory::buffer_view(
        reinterpret_cast<void*>(desc.address()), desc.bytes(), srf::codable::decode_memory_type(desc.memory_kind()));
}

void EncodedObject::copy_from_registered_buffer(const codable::idx_t& idx, srf::memory::buffer_view& dst_view) const
{
    const auto& remote = proto().descriptors().at(idx).remote_desc();
    CHECK_LE(dst_view.bytes(), remote.bytes());

    // todo(ryan) - check locality, if we are on the same machine but a different instance, use direct method
    if (m_resources.network()->instance_id() == remote.instance_id())
    {
        LOG(FATAL) << "implement local copy";
    }
    else
    {
        // get endpoint to remote instance_id

        // determine if remote memory region is in the remote memory cache

        // if not, unpack the remote keys on the endpoint
        // - if marked as cacheable, register the entire remote region in the cache
        // - if not, add the remote keys to the request object do be deleted after the rdma get

        // issue the rdma get on the remote region targeting the local dst_view

        LOG(FATAL) << "implement remote copy";
    }
}
void EncodedObject::copy_from_eager_buffer(const codable::idx_t& idx, srf::memory::buffer_view& dst_view) const
{
    const auto& eager_buffer = proto().descriptors().at(idx).eager_desc();
    CHECK_LE(dst_view.bytes(), eager_buffer.data().size());

    if (dst_view.kind() == srf::memory::memory_kind::device)
    {
        LOG(FATAL) << "implement async device copies";
    }

    if (dst_view.kind() == srf::memory::memory_kind::none)
    {
        LOG(WARNING) << "got a memory::kind::none";
    }
    std::memcpy(dst_view.data(), eager_buffer.data().data(), dst_view.bytes());
}
}  // namespace srf::internal::remote_descriptor

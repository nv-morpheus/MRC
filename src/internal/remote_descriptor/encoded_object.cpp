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

    if (!ucx_block && !force_register && view.bytes() < 1_MiB)
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
    encode_descriptor(*desc, view, *ucx_block, should_cache);
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

void EncodedObject::encode_descriptor(codable::protos::RemoteMemoryDescriptor& desc,
                                      srf::memory::const_buffer_view view,
                                      const ucx::MemoryBlock& ucx_block,
                                      bool should_cache)
{
    desc.set_memory_region_addr(reinterpret_cast<std::uint64_t>(ucx_block.data()));
    desc.set_memory_region_bytes(ucx_block.bytes());
    desc.set_remote_address(reinterpret_cast<std::uint64_t>(view.data()));
    desc.set_remote_bytes(view.bytes());
    desc.set_memory_kind(srf::codable::encode_memory_type(view.kind()));
    desc.set_remote_key(ucx_block.packed_remote_keys());
    desc.set_should_cache(should_cache);
}

srf::memory::buffer_view EncodedObject::decode_descriptor(const codable::protos::RemoteMemoryDescriptor& desc)
{
    return srf::memory::buffer_view(reinterpret_cast<void*>(desc.remote_address()),
                                    desc.remote_bytes(),
                                    srf::codable::decode_memory_type(desc.memory_kind()));
}

}  // namespace srf::internal::remote_descriptor

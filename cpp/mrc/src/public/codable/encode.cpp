/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/codable/encode.hpp"

#include "mrc/memory/literals.hpp"  // IWYU pragma: keep

namespace mrc::codable {

using namespace mrc::memory::literals;

EncoderBase::EncoderBase(LocalSerializedWrapper& encoded_object, memory::memory_block_provider& block_provider) :
  m_encoded_object(encoded_object),
  m_block_provider(block_provider)
{}

size_t EncoderBase::write_descriptor(memory::const_buffer_view view, DescriptorKind kind)
{
    if (kind == DescriptorKind::Default)
    {
        // Choose which one based on size
        if (view.bytes() < 64_KiB)
        {
            return this->write_descriptor(view, DescriptorKind::Eager);
        }

        return this->write_descriptor(view, DescriptorKind::Deferred);
    }

    protos::MemoryDescriptor& desc = m_encoded_object.add_descriptor();

    auto desc_idx = m_encoded_object.descriptors_size() - 1;

    switch (kind)
    {
    case DescriptorKind::Eager: {
        auto* eager_desc = desc.mutable_eager_desc();

        eager_desc->set_data(view.data(), view.bytes());

        return desc_idx;
    }
    case DescriptorKind::Deferred: {
        // Add a payload
        auto* payload = m_encoded_object.proto().add_payloads();

        payload->set_address(reinterpret_cast<uintptr_t>(view.data()));
        payload->set_bytes(view.bytes());
        payload->set_memory_kind(encode_memory_type(view.kind()));

        auto* deferred_desc = desc.mutable_deferred_desc();

        deferred_desc->set_payload_idx(m_encoded_object.payloads().size() - 1);

        return desc_idx;
    }
    default:
        throw std::runtime_error("Unknown descriptor kind");
    }
}
}  // namespace mrc::codable

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

#include "mrc/codable/decode.hpp"

namespace mrc::codable {

DecoderBase::DecoderBase(const LocalSerializedWrapper& encoded_object) : m_encoded_object(encoded_object) {}

void DecoderBase::read_descriptor(size_t offset, memory::buffer_view dst_view) const
{
    // Find the descriptor from the current object and offset
    const auto& descriptor = m_encoded_object.get_descriptor_from_offset(offset);

    if (descriptor.has_eager_desc())
    {
        const auto& eager_desc = descriptor.eager_desc();

        CHECK_EQ(eager_desc.memory_kind(), mrc::codable::protos::MemoryKind::Host) << "Only host memory is "
                                                                                      "supported at this time";

        // Finally, copy the data
        std::memcpy(dst_view.data(), eager_desc.data().data(), eager_desc.data().size());
    }
    else
    {
        // Its deferred. Find the corresponding payload object
        const auto& deferred_desc = descriptor.deferred_desc();

        // Find the payload object
        const auto& payload = m_encoded_object.payloads().at(deferred_desc.payload_idx());

        CHECK_EQ(payload.memory_kind(), mrc::codable::protos::MemoryKind::Host) << "Only host memory is "
                                                                                   "supported at this time";

        // Finally, copy the data
        std::memcpy(dst_view.data(), reinterpret_cast<const void*>(payload.address()), payload.bytes());
    }
}

std::size_t DecoderBase::descriptor_size(size_t offset) const
{
    // Find the descriptor from the current object and offset
    const auto& descriptor = m_encoded_object.get_descriptor_from_offset(offset);

    if (descriptor.has_eager_desc())
    {
        return descriptor.eager_desc().data().size();
    }

    if (descriptor.has_deferred_desc())
    {
        // Its deferred. Find the corresponding payload object
        const auto& deferred_desc = descriptor.deferred_desc();

        // Find the payload object
        const auto& payload = m_encoded_object.payloads().at(deferred_desc.payload_idx());

        return payload.bytes();
    }

    throw std::runtime_error("Invalid descriptor. Does not have either an eager or deferred descriptor");
}

}  // namespace mrc::codable

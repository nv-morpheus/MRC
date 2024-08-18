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

#include <cuda_runtime.h>

namespace mrc::codable {

DecoderBase::DecoderBase(const DescriptorObjectHandler& encoded_object) : m_encoded_object(encoded_object) {}

void DecoderBase::read_descriptor(memory::buffer_view dst_view) const
{
    // Get the next unprocessed payload
    const auto& payload = m_encoded_object.get_current_payload();

    if (payload.has_eager_msg())
    {
        const auto& eager_msg = payload.eager_msg();

        // Eager messages will always be host to host memcopy
        std::memcpy(dst_view.data(), eager_msg.data().data(), eager_msg.data().size());
    }
    else
    {
        const auto& deferred_msg = payload.deferred_msg();

        // Depending on the message memory type, we will use a different memcpy method to properly copy the data
        switch (payload.memory_kind())
        {
            case protos::MemoryKind::Host:
            case protos::MemoryKind::Pinned:
                std::memcpy(dst_view.data(), reinterpret_cast<const void*>(deferred_msg.address()), deferred_msg.bytes());
                break;
            case protos::MemoryKind::Device:
                cudaMemcpy(dst_view.data(), reinterpret_cast<const void*>(deferred_msg.address()), deferred_msg.bytes(), cudaMemcpyDeviceToHost);
                break;
            case protos::MemoryKind::Managed:
                cudaMemcpy(dst_view.data(), reinterpret_cast<const void*>(deferred_msg.address()), deferred_msg.bytes(), cudaMemcpyDefault);
            default:
                LOG(FATAL) << "unhandled protos::MemoryKind";
        };
    }

    m_encoded_object.increment_payload_idx();
}

std::size_t DecoderBase::descriptor_size() const
{
    // Get the next unprocessed payload
    const auto& payload = m_encoded_object.get_current_payload();

    return payload.has_eager_msg() ? payload.eager_msg().data().size() : payload.deferred_msg().bytes();

    throw std::runtime_error("Invalid descriptor. Does not have either an eager or deferred descriptor");
}

}  // namespace mrc::codable

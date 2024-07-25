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

EncoderBase::EncoderBase(DescriptorObjectHandler& encoded_object) :
  m_encoded_object(encoded_object)
{}

void EncoderBase::write_descriptor(memory::const_buffer_view view)
{
    MessageKind kind = (view.bytes() < 64_KiB) ? MessageKind::Eager : MessageKind::Deferred;

    protos::Payload* payload = m_encoded_object.proto().add_payloads();
    payload->set_memory_kind(encode_memory_type(view.kind()));

    switch (kind)
    {
    case MessageKind::Eager: {
        if (view.kind() == memory::memory_kind::host)
        {
            auto* eager_msg = payload->mutable_eager_msg();

            eager_msg->set_data(view.data(), view.bytes());

            return;
        }
    }
    case MessageKind::Deferred: {
        auto* deferred_msg = payload->mutable_deferred_msg();

        deferred_msg->set_address(reinterpret_cast<uintptr_t>(view.data()));
        deferred_msg->set_bytes(view.bytes());

        return;
    }
    default:
        throw std::runtime_error("Unknown descriptor kind");
    }
}
}  // namespace mrc::codable

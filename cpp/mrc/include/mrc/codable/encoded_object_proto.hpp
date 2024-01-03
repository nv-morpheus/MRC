/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "mrc/codable/api.hpp"
#include "mrc/memory/buffer.hpp"
#include "mrc/memory/buffer_view.hpp"
#include "mrc/memory/memory_block.hpp"

#include <memory>

namespace mrc::codable {

class EncodedObjectProto
{
  public:
    EncodedObjectProto() = default;

    bool operator==(const EncodedObjectProto& other) const;

    size_t objects_size() const;
    size_t descriptors_size() const;

    bool context_acquired() const;

    obj_idx_t push_context(std::type_index type_index);

    void pop_context(obj_idx_t object_idx);

    // Adds an eager descriptor and copies the data into the protobuf
    idx_t add_eager_descriptor(memory::const_buffer_view view);

    // Adds a remote memory descriptor and sets the properties
    idx_t add_remote_memory_descriptor(uint64_t instance_id,
                                       uintptr_t address,
                                       size_t bytes,
                                       uintptr_t memory_block_address,
                                       size_t memory_block_size,
                                       void* remote_key,
                                       memory::memory_kind memory_kind);

    memory::buffer to_bytes(std::shared_ptr<memory::memory_resource> mr) const;

    memory::buffer_view to_bytes(memory::buffer_view buffer) const;

    static std::unique_ptr<EncodedObjectProto> from_bytes(memory::const_buffer_view view);

  private:
    mrc::codable::protos::EncodedObject m_proto;

    bool m_context_acquired{false};
    mutable std::mutex m_mutex;

    std::optional<obj_idx_t> m_parent{std::nullopt};
};

class EncodedObjectWithPayload : public EncodedObjectProto
{
  private:
    std::vector<memory::memory_block> m_blocks;
};

}  // namespace mrc::codable

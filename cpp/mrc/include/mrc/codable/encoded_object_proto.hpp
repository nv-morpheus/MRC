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
#include "mrc/protos/codable.pb.h"

#include <memory>
#include <optional>
#include <stack>
#include <typeindex>

namespace mrc::codable {

struct SerializedItem
{
    int32_t starting_descriptor_idx;
    int32_t parent_object_idx;
    uint64_t type_index_hash;
};

struct SerializedMemoryBlocks
{
    memory::const_buffer_view buffer;
    bool is_eager;
};

class LocalSerializedWrapper
{
  public:
    LocalSerializedWrapper() = default;

    bool operator==(const LocalSerializedWrapper& other) const;

    mrc::codable::protos::LocalSerializedObject& proto();

    const mrc::codable::protos::LocalSerializedObject& proto() const;

    bool has_current_object() const;
    void reset_current_object_idx();
    size_t get_current_object_idx() const;
    size_t push_current_object_idx(std::type_index type_info);
    size_t push_current_object_idx(std::type_index type_info) const;
    void pop_current_object_idx(size_t pushed_object_idx);
    void pop_current_object_idx(size_t pushed_object_idx) const;

    size_t objects_size() const;
    size_t descriptors_size() const;
    size_t payloads_size() const;

    protos::Object& get_object(size_t idx);
    const protos::Object& get_object(size_t idx) const;

    size_t get_descriptor_idx(size_t object_idx, size_t desc_offset) const;

    protos::MemoryDescriptor& get_descriptor(size_t desc_idx);
    const protos::MemoryDescriptor& get_descriptor(size_t desc_idx) const;

    protos::MemoryDescriptor& get_descriptor_from_offset(size_t desc_offset,
                                                         std::optional<size_t> object_idx = std::nullopt);
    const protos::MemoryDescriptor& get_descriptor_from_offset(size_t desc_offset,
                                                               std::optional<size_t> object_idx = std::nullopt) const;

    protos::LocalPayload& get_payload(size_t idx);
    const protos::LocalPayload& get_payload(size_t idx) const;

    // protos::Object& add_object();
    size_t add_object(std::type_index type_info);

    protos::MemoryDescriptor& add_descriptor();
    size_t add_descriptor(memory::const_buffer_view view, DescriptorKind kind);

    // bool context_acquired() const;

    // obj_idx_t push_context(std::type_index type_index);

    // void pop_context(obj_idx_t object_idx);

    // obj_idx_t push_decode_context(std::type_index type_index) const;

    // void pop_decode_context(obj_idx_t object_idx) const;

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

    // /**
    //  * @brief Hash of std::type_index for the object at idx
    //  *
    //  * @param object_idx
    //  * @return std::type_index
    //  */
    // std::size_t type_index_hash_for_object(const obj_idx_t& object_idx) const;

    // /**
    //  * @brief Starting index of object at idx
    //  *
    //  * @param object_idx
    //  * @return idx_t
    //  */
    // idx_t start_idx_for_object(const obj_idx_t& object_idx) const;

    // /**
    //  * @brief Parent for object at idx
    //  *
    //  * @return std::optional<obj_idx_t> - if nullopt, then the object is a top-level object; otherwise, it is a child
    //  * object with a parent object at the returned value
    //  */
    // std::optional<obj_idx_t> parent_obj_idx_for_object(const obj_idx_t& object_idx) const;

    const mrc::codable::protos::SerializedInfo& info() const;

    mrc::codable::protos::SerializedInfo& mutable_info();

    const ::google::protobuf::RepeatedPtrField<::mrc::codable::protos::LocalPayload>& payloads() const;

    memory::buffer to_bytes(std::shared_ptr<memory::memory_resource> mr) const;

    memory::buffer_view to_bytes(memory::buffer_view buffer) const;

    static std::unique_ptr<LocalSerializedWrapper> from_bytes(memory::const_buffer_view view);

  private:
    mrc::codable::protos::LocalSerializedObject m_proto;

    // std::vector<SerializedItem> m_objects;
    // std::vector<SerializedMemoryBlocks> m_descriptors;

    // bool m_context_acquired{false};
    // mutable std::mutex m_mutex;

    mutable size_t m_object_counter;  // Tracks the number of times "push_current_object_idx" has been called
    mutable std::stack<size_t> m_object_idx_stack;

    // mutable std::optional<size_t> m_current_obj_idx{std::nullopt};
};

}  // namespace mrc::codable

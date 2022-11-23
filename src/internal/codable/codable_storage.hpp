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

#pragma once

#include "internal/codable/decodable_storage_view.hpp"
#include "internal/codable/storage_view.hpp"
#include "internal/resources/forward.hpp"
#include "internal/ucx/forward.hpp"

#include "mrc/codable/api.hpp"
#include "mrc/memory/buffer.hpp"
#include "mrc/memory/buffer_view.hpp"
#include "mrc/protos/codable.pb.h"
#include "mrc/types.hpp"

#include <google/protobuf/message.h>

#include <cstddef>
#include <map>
#include <mutex>
#include <optional>
#include <typeindex>
#include <vector>

namespace mrc::internal::codable {

/**
 * @brief CodableStorage implements both the IEncodableStorage and the IDecodableStorage interfaces
 *
 *
 *
 */
class CodableStorage final : public mrc::codable::ICodableStorage, public StorageView, public DecodableStorageView
{
  public:
    CodableStorage(resources::PartitionResources& resources);
    CodableStorage(mrc::codable::protos::EncodedObject proto, resources::PartitionResources& resources);

    mrc::codable::IEncodableStorage& encodable();

    mrc::codable::IDecodableStorage& decodable();

  private:
    mrc::codable::protos::EncodedObject& mutable_proto() final;

    bool context_acquired() const final;

    obj_idx_t push_context(std::type_index type_index) final;

    void pop_context(obj_idx_t object_idx) final;

    // register memory region
    // may return nullopt if the region is considered too small
    std::optional<idx_t> register_memory_view(mrc::memory::const_buffer_view view, bool force_register = false) final;

    // copy to eager descriptor
    idx_t copy_to_eager_descriptor(mrc::memory::const_buffer_view view) final;

    // add a meta_data descriptor
    idx_t add_meta_data(const google::protobuf::Message& meta_data) final;

    // create a buffer owned by this
    idx_t create_memory_buffer(std::size_t bytes) final;

    // copy data to a created buffer
    void copy_to_buffer(idx_t buffer_idx, mrc::memory::const_buffer_view view) final;

    // get a mutable view into the memory of a descriptor
    mrc::memory::buffer_view mutable_host_buffer_view(const idx_t& buffer_idx) final;

    /**
     * @brief Converts a memory block to a RemoteMemoryDescriptor proto
     */
    static void encode_descriptor(const InstanceID& instance_id,
                                  mrc::codable::protos::RemoteMemoryDescriptor& desc,
                                  mrc::memory::const_buffer_view view,
                                  const ucx::MemoryBlock& ucx_block,
                                  bool should_cache = false);

    static mrc::memory::buffer_view decode_descriptor(const mrc::codable::protos::RemoteMemoryDescriptor& desc);

    mrc::codable::protos::EncodedObject& get_mutable_proto();

    const mrc::codable::protos::EncodedObject& get_proto() const final;

    resources::PartitionResources& resources() const final;

    resources::PartitionResources& m_resources;
    mrc::codable::protos::EncodedObject m_proto;
    std::map<idx_t, mrc::memory::buffer> m_buffers;
    std::vector<mrc::memory::const_buffer_view> m_temporary_registrations;
    std::optional<obj_idx_t> m_parent{std::nullopt};
    bool m_context_acquired{false};
    mutable std::mutex m_mutex;
};

}  // namespace mrc::internal::codable

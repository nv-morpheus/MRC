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

#include "srf/codable/codable_protocol.hpp"
#include "srf/exceptions/runtime_error.hpp"
#include "srf/memory/buffer.hpp"
#include "srf/memory/buffer_view.hpp"
#include "srf/protos/codable.pb.h"
#include "srf/utils/macros.hpp"

#include <glog/logging.h>
#include <google/protobuf/message.h>

#include <cstddef>
#include <map>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

namespace srf::codable {

/**
 * @brief Defines the sequence of memory regions (blobs) and meta data to encode an object.
 *
 * EncodedObject is a non-owning sequence of descriptors which represent the encoding/serialization of an object. While
 * EncodedObject does not own the object for which it is encoding, it may own memory regions/buffers for memory regions
 * which require external copies from the original object.
 *
 * Typically, a derived Encoded<T> is passed to codable_protocol::serialize(t) which required that the process's
 * environment be configured with the SRF Runtime. The SRF Runtime is instantiated on all threads provided by the
 * Executor.
 *
 * @note The serialization of an object should create one, and only one, ContextGuard by calling the
 * acquire_encoding_context method from the derived Encoded<T>.
 */
class EncodedObject
{
  public:
    /**
     * @brief ObjectDescriptor describing the encoded object.
     * @return const protos::ObjectDescriptor&
     */
    const protos::EncodedObject& proto() const;

    /**
     * @brief Access const memory::buffer_view of the RemoteMemoryDescriptor at the required index
     * @return memory::const_buffer_view
     */
    memory::const_buffer_view memory_block(std::size_t idx) const;

    /**
     * @brief Decode meta data associated the MetaDataDescriptor at the requested index.
     *
     * @tparam MetaDataT
     * @return MetaDataT
     */
    template <typename MetaDataT>
    MetaDataT meta_data(std::size_t idx) const;

    /**
     * @brief
     *
     * @return protos::EagerDescriptor&
     */
    const protos::EagerDescriptor& eager_descriptor(std::size_t idx) const;

    /**
     * @brief The number of unique memory regions contained in the multiple part descriptor.
     * @return std::size_t
     */
    std::size_t descriptor_count() const;

    /**
     * @brief The number of unqiue objects described by the encoded object
     * @return std::size_t
     */
    std::size_t object_count() const;

    /**
     * @brief Hash of std::type_index for the object at idx
     *
     * @param object_idx
     * @return std::type_index
     */
    std::size_t type_index_hash_for_object(std::size_t object_idx) const;

    /**
     * @brief Starting index of object at idx
     *
     * @param object_idx
     * @return std::size_t
     */
    std::size_t start_idx_for_object(std::size_t object_idx) const;

  protected:
    /**
     * @brief Access a mutable const_buffer_view at the requested index
     *
     * @param idx
     * @return memory::const_buffer_view
     */
    memory::buffer_view mutable_memory_block(std::size_t idx) const;

    /**
     * @brief Converts a memory block to a RemoteMemoryDescriptor proto
     *
     * @param view
     * @return protos::RemoteMemoryDescriptor
     */
    static protos::RemoteMemoryDescriptor encode_descriptor(memory::const_buffer_view view, std::string keys);

    /**
     * @brief Converts a RemoteMemoryDescriptor proto to a mutable memory block
     *
     * @param desc
     * @return memory::buffer_view
     */
    static memory::buffer_view decode_descriptor(const protos::RemoteMemoryDescriptor& desc);

    /**
     * @brief Add a custom protobuf meta data to the descriptor list
     *
     * @param meta_data
     * @return std::size_t
     */
    std::size_t add_meta_data(const google::protobuf::Message& meta_data);

    /**
     * @brief Add a blob view to the sequence of descriptors
     *
     * @param view
     * @param meta_data
     * @return std::size_t
     */
    std::size_t add_memory_block(memory::const_buffer_view view);

    /**
     * @brief Add a buffer, owned by EncodedObject, that can be used to hold a contiguous block of data.
     *
     * After creation, the const_buffer_view can be accessed by calling view with the index returned.
     *
     * @note The memory_resource backing the creation of the buffer<> comes from the SRF Runtime's thread local resource
     * object.
     *
     * @param bytes
     * @param meta_data
     * @return std::size_t
     */
    std::size_t add_host_buffer(std::size_t bytes);

    /**
     * @brief Add a buffer, owned by EncodedObject, that can be used to hold a contiguous block of data.
     *
     * After creation, the const_buffer_view can be accessed by calling view on the index returned.
     *
     * @note The memory_resource backing the creation of the buffer<> comes from the SRF Runtime's thread local resource
     * object.
     *
     * @param bytes
     * @param meta_data
     * @return std::size_t
     */
    std::size_t add_device_buffer(std::size_t bytes);

    /**
     * @brief Add an eager buffer owned by EncodedObject. This buffer will be serialized and sent as part of the control
     * message.
     *
     * Eager buffers are limited in size by compilation limits, SRF_MAX_EAGER_BUFFER_SIZE
     *
     * @return std::size_t
     */
    std::size_t add_eager_buffer(const void* data, std::size_t bytes);

    /**
     * @brief Basic guard object that must be acquried before being able to access the add_* or mutable_* methods
     */
    class ContextGuard final
    {
      public:
        ContextGuard(EncodedObject& encoded_object, std::type_index type_index);
        ~ContextGuard();

        DELETE_COPYABILITY(ContextGuard);

      private:
        EncodedObject& m_encoded_object;
    };

  private:
    /**
     * @brief Add a buffer
     *
     * @tparam PropertiesT
     * @param view
     * @param bytes
     * @return std::size_t
     */
    std::size_t add_buffer(memory::buffer&& buffer);

    /**
     * @brief Used to push a Object message with the starting descriptor index and type_index to the main proto
     * @param type_index
     */
    void add_type_index(std::type_index type_index);

    protos::EncodedObject m_proto;
    std::map<std::size_t, memory::buffer> m_buffers;
    std::vector<std::pair<int, std::type_index>> m_object_info;  // typeindex and starting descriptor index
    bool m_context_acquired{false};
    friend ContextGuard;
};

/**
 * @brief Used by codable_protocol to create a unique_ptr<EncodedObject> from an instance of ObjectT
 *
 * @note Encoded is final, but provides codable_protocol access to the protected methods of EncodedObject via
 * friendship.
 *
 * @tparam ObjectT
 */
template <typename T>
class Encoded final : public EncodedObject
{
    friend T;
    friend codable_protocol<T>;

    [[nodiscard]] ContextGuard acquire_encoding_context()
    {
        return ContextGuard(*this, std::type_index(typeid(T)));
    }
};

// Implementation

template <typename MetaDataT>
MetaDataT EncodedObject::meta_data(std::size_t idx) const
{
    DCHECK_LT(idx, descriptor_count());
    const auto& desc = m_proto.descriptors().at(idx);
    CHECK(desc.has_meta_data_desc());

    MetaDataT meta_data;
    auto ok = desc.meta_data_desc().meta_data().UnpackTo(&meta_data);
    if (!ok)
    {
        throw exceptions::SrfRuntimeError("unable to decode meta data to the requestd message type");
    }
    return meta_data;
}

}  // namespace srf::codable

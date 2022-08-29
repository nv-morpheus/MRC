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
#include "srf/codable/memory.hpp"
#include "srf/codable/types.hpp"
#include "srf/exceptions/runtime_error.hpp"
#include "srf/memory/buffer.hpp"
#include "srf/memory/buffer_view.hpp"
#include "srf/memory/memory_kind.hpp"
#include "srf/memory/resources/memory_resource.hpp"
#include "srf/protos/codable.pb.h"
#include "srf/types.hpp"
#include "srf/utils/macros.hpp"

#include <glog/logging.h>
#include <google/protobuf/message.h>

#include <cstddef>
#include <map>
#include <memory>
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
 * Typically, a derived Encoded<T> is passed to codable_protocol::serialize(t). An EncodedObject can be constructed from
 * the Runtime::make_encoded_object() method which will use the memory and registration cache from the respective
 * partition.
 *
 * @note The serialization of an object should create one, and only one, ContextGuard by calling the
 * acquire_encoding_context method from the derived Encoded<T>.
 */
class EncodedObject
{
  public:
    EncodedObject() = default;
    EncodedObject(protos::EncodedObject proto);
    virtual ~EncodedObject() = default;

    /**
     * @brief ObjectDescriptor describing the encoded object.
     * @return const protos::ObjectDescriptor&
     */
    const protos::EncodedObject& proto() const;

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
    std::size_t type_index_hash_for_object(const obj_idx_t& object_idx) const;

    /**
     * @brief Starting index of object at idx
     *
     * @param object_idx
     * @return std::size_t
     */
    idx_t start_idx_for_object(const obj_idx_t& object_idx) const;

  protected:
    protos::EncodedObject& proto();

    const bool& context_acquired() const;

    /**
     * @brief Converts a memory block to a RemoteMemoryDescriptor proto
     *
     * @param view
     * @return protos::RemoteMemoryDescriptor
     */
    // move to implementation detail
    // static protos::RemoteMemoryDescriptor encode_descriptor(memory::const_buffer_view view, std::string keys);

    /**
     * @brief Converts a RemoteMemoryDescriptor proto to a mutable memory block
     *
     * @param desc
     * @return memory::buffer_view
     */
    // move to implementation detail
    // static memory::buffer_view decode_descriptor(const protos::RemoteMemoryDescriptor& desc);

    /**
     * @brief Add a view to the descriptor list
     *
     * This is a view into the the memory of the object which is being encoded. This buffer is not owned by this
     * EncodedObject. This method requires the SRF runtime to provide the UCX remote keys into the remote memory
     * descriptor.
     *
     * @param view
     * @param force_register - register the block even if it is smaller than the minimum suggested block size
     * @return std::optional<idx_t>
     */
    virtual std::optional<idx_t> register_memory_view(memory::const_buffer_view view, bool force_register = false) = 0;

    /**
     * @brief Add an eager buffer owned by EncodedObject. This buffer will be serialized and sent as part of the control
     * message.
     *
     * Eager buffers are limited in size by compilation limits, SRF_MAX_EAGER_BUFFER_SIZE
     *
     * @return std::size_t
     */
    virtual idx_t copy_to_eager_descriptor(memory::const_buffer_view view) = 0;

    /**
     * @brief Add a custom protobuf meta data to the descriptor list
     *
     * @param meta_data
     * @return std::size_t
     */
    std::size_t add_meta_data(const google::protobuf::Message& meta_data);

    /**
     * @brief Creates a memory::buffer, owned by EncodedObject, that can be used to hold a contiguous block of data.
     *
     * After creation, the const_buffer_view can be accessed by calling view with the index returned.
     *
     * @note The memory_resource backing the creation of the buffer<> comes from the SRF Runtime's thread local resource
     * object.
     *
     * @param bytes
     * @return idx_t
     */
    virtual idx_t create_memory_buffer(std::size_t bytes) = 0;

    /**
     * @brief Access a mutable const_buffer_view at the requested index
     *
     * @param idx
     * @return memory::const_buffer_view
     */
    virtual memory::buffer_view mutable_memory_buffer(const idx_t& idx) const = 0;

    // DECODE operations

    virtual void copy_from_buffer(const idx_t& idx, memory::buffer_view dst_view) const = 0;

    /**
     * @brief Decode meta data associated the MetaDataDescriptor at the requested index.
     *
     * Provides an unpacked MetaDataT protobuf message to the caller.
     *
     * @tparam MetaDataT
     * @return MetaDataT
     */
    template <typename MetaDataT>
    MetaDataT meta_data(const idx_t& idx) const
    {
        DCHECK_LT(idx, descriptor_count());
        const auto& desc = proto().descriptors().at(idx);
        CHECK(desc.has_meta_data_desc());

        MetaDataT meta_data;
        auto ok = desc.meta_data_desc().meta_data().UnpackTo(&meta_data);
        if (!ok)
        {
            throw exceptions::SrfRuntimeError("unable to decode meta data to the requestd message type");
        }
        return meta_data;
    }

    /**
     * @brief Size in bytes of a given descriptor index
     *
     * @param idx
     * @return std::size_t
     */
    std::size_t buffer_size(const idx_t& idx) const;

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
class EncodableObject final : public EncodedObject
{
    friend T;
    friend codable_protocol<T>;

    [[nodiscard]] ContextGuard acquire_encoding_context()
    {
        return ContextGuard(*this, std::type_index(typeid(T)));
    }
};

template <typename T>
class DecodableObject : public EncodedObject
{
    friend T;
    friend codable_protocol<T>;
};

}  // namespace srf::codable

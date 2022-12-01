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

#include "mrc/codable/codable_protocol.hpp"
#include "mrc/codable/types.hpp"
#include "mrc/memory/buffer_view.hpp"

#include <google/protobuf/message.h>

#include <optional>
#include <typeindex>

namespace mrc::codable {

namespace protos {
class EncodedObject;
}

class IStorage;
class IEncodableStorage;
class IDecodableStorage;

template <typename T>
class Encoder;

template <typename T>
class Decoder;

class EncodingOptions;

/**
 * @brief Interface for an EncodedObject
 *
 * Defines the sequence of memory regions (blobs) and meta data to encode one or more objects.
 *
 * EncodedObject is a non-owning sequence of descriptors which represent the encoding/serialization of an object. While
 * EncodedObject does not own the object for which it is encoding, it may own memory regions/buffers for memory regions
 * which require external copies from the original object.
 *
 * Typically, a derived EncodableObject<T> is passed to codable_protocol::serialize(t). An EncodedObject can be
 * constructed from the Runtime::make_encoded_object() method which will use the memory and registration cache from the
 * respective partition.
 *
 * @note The serialization of an object should create one, and only one, Context by calling the
 * acquire_encoding_context method from the derived Encoded<T>.
 */
class IStorage
{
  public:
    using idx_t     = mrc::codable::idx_t;
    using obj_idx_t = mrc::codable::obj_idx_t;

    virtual ~IStorage() = default;

    /**
     * @brief ObjectDescriptor describing the encoded object.
     * @return const protos::ObjectDescriptor&
     */
    virtual const protos::EncodedObject& proto() const = 0;

    /**
     * @brief The number of unqiue objects described by the encoded object
     * @return std::size_t
     */
    virtual obj_idx_t object_count() const = 0;

    /**
     * @brief The number of unique memory regions contained in the multiple part descriptor.
     * @return std::size_t
     */
    virtual idx_t descriptor_count() const = 0;

    /**
     * @brief Hash of std::type_index for the object at idx
     *
     * @param object_idx
     * @return std::type_index
     */
    virtual std::size_t type_index_hash_for_object(const obj_idx_t& object_idx) const = 0;

    /**
     * @brief Starting index of object at idx
     *
     * @param object_idx
     * @return idx_t
     */
    virtual idx_t start_idx_for_object(const obj_idx_t& object_idx) const = 0;

    /**
     * @brief Parent for object at idx
     *
     * @return std::optional<obj_idx_t> - if nullopt, then the object is a top-level object; otherwise, it is a child
     * object with a parent object at the returned value
     */
    virtual std::optional<obj_idx_t> parent_obj_idx_for_object(const obj_idx_t& object_idx) const = 0;
};

class IEncodableStorage : public virtual IStorage
{
  public:
    ~IEncodableStorage() override = default;

  protected:
    /**
     * @brief Add a view to the descriptor list
     *
     * This is a view into the the memory of the object which is being encoded. The memory of the view is not managed by
     * the EncodedObject. This call will attempt to register the region of memory with the NIC. If the memory region is
     * smaller than the runtime limit or if the memory is unable to be registerd, the method will return a nullopt;
     * otherwise, the index to the internal descriptor will be returned.
     *
     * @param view
     * @param force_register - register the block even if it is smaller than the minimum suggested block size
     * @return std::optional<idx_t>
     */
    [[nodiscard]] virtual std::optional<idx_t> register_memory_view(memory::const_buffer_view view,
                                                                    bool force_register = false) = 0;

    /**
     * @brief Add an eager buffer owned by EncodedObject. This buffer will be serialized and sent as part of the control
     * message.
     *
     * Eager buffers are limited in size by compilation limits, MRC_MAX_EAGER_BUFFER_SIZE and MRC_MAX_EAGER_TOTAL_BYTES
     * TODO(ryan) - enforce these compile time limits
     *
     * @return std::size_t
     */
    virtual idx_t copy_to_eager_descriptor(memory::const_buffer_view view) = 0;

    /**
     * @brief Add a custom protobuf meta data to the descriptor list
     *
     * @param meta_data
     * @return idx_t
     */
    virtual idx_t add_meta_data(const google::protobuf::Message& meta_data) = 0;

    /**
     * @brief Creates a memory::buffer, owned by EncodedObject, that can be used to hold a contiguous block of data.
     *
     * After creation, the const_buffer_view can be accessed by calling view with the index returned.
     *
     * @note The memory_resource backing the creation of the buffer<> comes from the MRC Runtime's thread local resource
     * object.
     *
     * @param bytes
     * @return idx_t
     */
    virtual idx_t create_memory_buffer(std::size_t bytes) = 0;

    /**
     * @brief Copy data to the data represented by a descriptor
     *
     * @param buffer_idx
     * @param view
     */
    virtual void copy_to_buffer(idx_t buffer_idx, memory::const_buffer_view view) = 0;

    /**
     * @brief Provide a mutable host buffer view into a descriptor.
     * @note The descriptor must be associated with host memory, not device memory
     */
    virtual memory::buffer_view mutable_host_buffer_view(const idx_t& buffer_idx) = 0;

    /**
     * @brief Returns true if an external entity is holding a context; otherwise, false.
     *
     * @return true
     * @return false
     */
    virtual bool context_acquired() const = 0;

    /**
     * @brief Push a Context on the stack
     *
     * @param type_index
     * @return Context
     */

  private:
    /**
     * @brief Mutable version of the backing protobuf storage
     *
     * @return protos::EncodedObject&
     */
    virtual protos::EncodedObject& mutable_proto() = 0;

    virtual obj_idx_t push_context(std::type_index type_index) = 0;
    virtual void pop_context(obj_idx_t object_idx)             = 0;

    template <typename T>
    friend class Encoder;
};

class IDecodableStorage : public virtual IStorage
{
  public:
    ~IDecodableStorage() override = default;

  protected:
    virtual void copy_from_buffer(const idx_t& idx, memory::buffer_view dst_view) const = 0;

    /**
     * @brief Size in bytes of a given descriptor index
     *
     * @param idx
     * @return std::size_t
     */
    virtual std::size_t buffer_size(const idx_t& idx) const = 0;

    /**
     * @brief Host Memory Resource
     *
     * @return std::shared_ptr<mrc::memory::memory_resource>
     */
    virtual std::shared_ptr<mrc::memory::memory_resource> host_memory_resource() const = 0;

    /**
     * @brief Device Memory Resource
     *
     * @return std::shared_ptr<mrc::memory::memory_resource>
     */
    virtual std::shared_ptr<mrc::memory::memory_resource> device_memory_resource() const = 0;

    template <typename T>
    friend class Decoder;
};

struct ICodableStorage : public virtual IEncodableStorage, public virtual IDecodableStorage
{};

}  // namespace mrc::codable

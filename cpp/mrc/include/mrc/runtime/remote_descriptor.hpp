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
#include "mrc/codable/decode.hpp"
#include "mrc/codable/encode.hpp"
#include "mrc/codable/encoded_object.hpp"
#include "mrc/codable/encoded_object_proto.hpp"
#include "mrc/memory/memory_block_provider.hpp"
#include "mrc/type_traits.hpp"  // IWYU pragma: keep
#include "mrc/types.hpp"
#include "mrc/utils/macros.hpp"

#include <glog/logging.h>

#include <any>
#include <cstddef>
#include <functional>
#include <memory>
#include <ostream>
#include <utility>

namespace mrc::remote_descriptor {
class Manager;
}  // namespace mrc::remote_descriptor

namespace mrc::resources {
class PartitionResources;
}  // namespace mrc::resources

namespace mrc::data_plane {
class DataPlaneResources2;
}

namespace mrc::runtime {

class Descriptor
{
  public:
    Descriptor();
    Descriptor(Descriptor&& other);
    Descriptor& operator=(Descriptor&& other);

    DELETE_COPYABILITY(Descriptor);

    virtual ~Descriptor() = default;

    // Returns true if this object can be cast to `Descriptor<T>`
    virtual bool is_typed() const
    {
        return false;
    }

    /**
     * @brief Returns true if this object is still connected to the global object; otherwise, ownership has been
     * transferred or released.
     *
     * @return true
     * @return false
     */
    virtual bool has_value() const = 0;

    /**
     * @brief Returns true if this object is still connected to the global object; otherwise, ownership has been
     * transferred or released.
     *
     * @return true
     * @return false
     */
    operator bool() const
    {
        return this->has_value();
    }

    // Releases ownership of the storage object returning it to the caller
    virtual std::unique_ptr<codable::IDecodableStorage> release_storage() = 0;

    /**
     * @brief Decode the globally accessible object into a local object T constructed from the partition resources which
     * currently owns the RemoteDescriptor.
     *
     * todo(mdemoret/ryanolson) - we should consider renaming this method to `await_decode` as this object may trigger
     * network operators and may yield the execution context.
     *
     * @tparam T
     * @param object_idx
     * @return T
     */
    template <typename T>
    T await_decode(std::size_t object_idx = 0);

  protected:
    std::unique_ptr<mrc::codable::ICodableStorage> make_storage() const;

    std::reference_wrapper<resources::PartitionResources> m_partition_resources;
};

class LocalDescriptor : public Descriptor
{};

template <typename T>
class TypedLocalDescriptor : public LocalDescriptor
{
  public:
    TypedLocalDescriptor(T&& value, bool eager = false) : m_has_value(true), m_value(std::move(value))
    {
        // Eagerly encode the value into a storage object if requested
        if (eager)
        {
            auto storage = this->make_storage();

            if constexpr (is_unique_ptr_v<T>)
            {
                mrc::codable::encode(*m_value, *storage);
            }
            else
            {
                mrc::codable::encode(m_value, *storage);
            }

            m_storage = std::move(storage);
        }
    }

    bool is_typed() const override
    {
        return true;
    }

    bool has_value() const override
    {
        return m_has_value;
    }

    std::unique_ptr<codable::IDecodableStorage> release_storage() override
    {
        CHECK(this->has_value()) << "Cannot get a storage from a Descriptor which has been released or transferred.";

        // Set the has value flag to false since we are releasing ownership of the storage object
        m_has_value = false;

        if (m_storage)
        {
            return std::move(m_storage);
        }

        // Otherwise create one lazily and return it
        auto storage = this->make_storage();

        if constexpr (is_unique_ptr_v<T>)
        {
            mrc::codable::encode(*m_value, *storage);
        }
        else
        {
            mrc::codable::encode(m_value, *storage);
        }

        return storage;
    }

    // Releases ownership of the value object returning it to the caller
    T release_value()
    {
        CHECK(this->has_value()) << "Cannot get a value from a Descriptor which has been released or transferred.";

        T tmp_value = std::move(m_value);

        // Set the has value flag to false since we are releasing ownership of the value object
        m_has_value = false;

        // Reset the storage just in case it was eager
        m_storage.reset();

        return std::move(tmp_value);
    }

  private:
    bool m_has_value{false};

    T m_value;
    std::unique_ptr<codable::IDecodableStorage> m_storage;
};

template <typename T>
class ResidentDescriptor final : public TypedLocalDescriptor<T>
{
  public:
    // Make sure we construct the storage lazily
    ResidentDescriptor(T&& value) : TypedLocalDescriptor<T>(std::move(value), false) {}
};

template <typename T>
class CodedDescriptor final : public TypedLocalDescriptor<T>
{
  public:
    // Make sure we construct the storage eagerly
    CodedDescriptor(T&& value) : TypedLocalDescriptor<T>(std::move(value), true) {}
};

/**
 * @brief Primary user-level object for interacting with globally accessible object.
 *
 * The RemoteDescriptor is an RAII object which manages the lifecycle of a globally accessible object held by the
 * RemoteDescriptor manager on a given instance of the MRC runtime.
 *
 * The RemoteDescriptor can be used to reconstruct the globally accessible object using the decode method. This may
 * trigger network operations.
 *
 * A RemoteDescriptor owns some number of reference counting tokens for the global object. The RemoteDescriptor may
 * release ownership of those tokens which would decrement the global reference count by the number of tokens held or it
 * may choose to transfer ownership of those tokens by transmitting this object across the data plane to be
 * reconstructed on a remote instance.
 *
 * When a RemoteDescriptor is tranferred, the resulting local RemoteDescriptor::has_value or bool operator returns
 * false, meaning it no longer has access to the global object.
 *
 */
class RemoteDescriptor final : public Descriptor
{
  public:
    RemoteDescriptor();
    ~RemoteDescriptor() override;

    RemoteDescriptor(RemoteDescriptor&& other) noexcept;
    RemoteDescriptor& operator=(RemoteDescriptor&& other) noexcept;

    bool has_value() const override;

    std::unique_ptr<codable::IDecodableStorage> release_storage() override;

  private:
    // RemoteDescriptor(std::shared_ptr<IRemoteDescriptorManager> manager,
    //                  std::unique_ptr<IRemoteDescriptorHandle> handle);

    RemoteDescriptor(std::unique_ptr<codable::IDecodableStorage> storage);

    // std::unique_ptr<IRemoteDescriptorHandle> release_handle();

    // std::shared_ptr<IRemoteDescriptorManager> m_manager;
    // std::unique_ptr<IRemoteDescriptorHandle> m_handle;

    std::unique_ptr<codable::IDecodableStorage> m_storage;

    friend remote_descriptor::Manager;
};

template <typename T>
T Descriptor::await_decode(std::size_t object_idx)
{
    if (this->is_typed())
    {
        // Short circuit cast to typed descriptor
        return static_cast<TypedLocalDescriptor<T>*>(this)->release_value();
    }

    // Gain access to the storage object
    auto storage = this->release_storage();

    return codable::Decoder<T>(*storage).deserialize(object_idx);
}

class LocalDescriptor2;
class RemoteDescriptor2;
class RemoteDescriptorImpl2;

class ValueDescriptor
{
  public:
    template <typename T>
    T release_value() &&;

  private:
    virtual std::unique_ptr<codable::DescriptorObjectHandler> encode() = 0;

    friend LocalDescriptor2;
};

struct TaggedDescriptor
{
    InstanceID destination;
    InstanceID source;
    std::unique_ptr<ValueDescriptor> descriptor;
};

template <typename T>
class TypedValueDescriptor : public ValueDescriptor
{
  public:
    const T& value() const
    {
        return m_value;
    }

    static std::unique_ptr<TypedValueDescriptor<T>> create(T&& value)
    {
        return std::unique_ptr<TypedValueDescriptor<T>>(new TypedValueDescriptor<T>(std::move(value)));
    }
    static std::unique_ptr<TypedValueDescriptor<T>> from_local(std::unique_ptr<LocalDescriptor2> local_descriptor);

  private:
    TypedValueDescriptor(T&& value) : m_value(std::move(value)) {}

    std::unique_ptr<codable::DescriptorObjectHandler> encode() override
    {
        return mrc::codable::encode2(m_value);
    }

    T m_value;

    friend class ValueDescriptor;
};

template <typename T>
T ValueDescriptor::release_value() &&
{
    auto typed_descriptor = dynamic_cast<TypedValueDescriptor<T>*>(this);

    if (!typed_descriptor)
    {
        LOG(FATAL) << "Cannot release value of type " << typeid(T).name() << " from descriptor of type "
                   << typeid(*this).name();
    }

    return std::move(typed_descriptor->m_value);
}

// Combines a EncodedObjectProto with a local registered buffers if needed
class LocalDescriptor2
{
  public:
    codable::DescriptorObjectHandler& encoded_object() const;

    static std::unique_ptr<LocalDescriptor2> from_value(std::unique_ptr<ValueDescriptor> value_descriptor,
                                                        std::shared_ptr<memory::memory_block_provider> block_provider);

    static std::unique_ptr<LocalDescriptor2> from_remote(std::unique_ptr<RemoteDescriptor2> remote_descriptor,
                                                         data_plane::DataPlaneResources2& data_plane_resources);

  private:
    LocalDescriptor2(std::unique_ptr<codable::DescriptorObjectHandler> encoded_object,
                     std::unique_ptr<ValueDescriptor> value_descriptor = nullptr);

    std::unique_ptr<codable::DescriptorObjectHandler> m_encoded_object;

    std::unique_ptr<ValueDescriptor> m_value_descriptor;  // Necessary to keep the value alive when serializing
};

class RemoteDescriptorImpl2
{
  public:
    codable::protos::DescriptorObject& encoded_object() const;

    memory::buffer to_bytes(std::shared_ptr<memory::memory_resource> mr) const;

    memory::buffer_view to_bytes(memory::buffer_view buffer) const;

    static std::shared_ptr<RemoteDescriptorImpl2> from_local(std::unique_ptr<LocalDescriptor2> local_desc,
                                                             data_plane::DataPlaneResources2& data_plane_resources);

    static std::shared_ptr<RemoteDescriptorImpl2> from_bytes(memory::const_buffer_view view);

  private:
    friend class RemoteDescriptor2;

    RemoteDescriptorImpl2(std::unique_ptr<codable::protos::DescriptorObject> encoded_object);

    std::unique_ptr<codable::protos::DescriptorObject> m_serialized_object;
};

class RemoteDescriptor2
{
  public:
    codable::protos::DescriptorObject& encoded_object() const;

    memory::buffer to_bytes(std::shared_ptr<memory::memory_resource> mr) const;

    memory::buffer_view to_bytes(memory::buffer_view buffer) const;

    static std::unique_ptr<RemoteDescriptor2> from_local(std::unique_ptr<LocalDescriptor2> local_desc,
                                                         data_plane::DataPlaneResources2& data_plane_resources);

    static std::unique_ptr<RemoteDescriptor2> from_bytes(memory::const_buffer_view view);

  private:
    RemoteDescriptor2(std::unique_ptr<codable::protos::DescriptorObject> encoded_object);

    RemoteDescriptor2(std::shared_ptr<RemoteDescriptorImpl2> impl);

    std::shared_ptr<RemoteDescriptorImpl2> m_impl;
};

template <typename T>
std::unique_ptr<TypedValueDescriptor<T>> TypedValueDescriptor<T>::from_local(
    std::unique_ptr<LocalDescriptor2> local_descriptor)
{
    // Reset the counter
    local_descriptor->encoded_object().reset_payload_idx();

    // Perform a decode to get the value
    return std::unique_ptr<TypedValueDescriptor<T>>(
        new TypedValueDescriptor<T>(mrc::codable::decode2<T>(local_descriptor->encoded_object())));
}

/**
 * @brief Descriptor2 class used to faciliate communication between any arbitrary pair of machines. Supports multi-node,
 * multi-gpu communication, and asynchronous data transfer.
 */
class Descriptor2
{
  public:
    /**
     * @brief Gets the protobuf object associated with this descriptor instance
     *
     * @return codable::protos::DescriptorObject&
     */
    virtual codable::protos::DescriptorObject& encoded_object();

    /**
     * @brief Serialize the encoded object stored by this descriptor into a byte stream for remote communication
     *
     * @param mr Instance of memory_resource for allocating a memory_buffer to return
     * @return memory::buffer
     */
    memory::buffer serialize(std::shared_ptr<memory::memory_resource> mr);

    /**
     * @brief Deserialize the encoded object stored by this descriptor into a class T instance
     *
     * @return T
     */
    template <typename T>
    [[nodiscard]] const T deserialize();

    /**
     * @brief Creates a Descriptor2 instance from a class T value
     *
     * @param value class T instance
     * @param data_plane_resources reference to DataPlaneResources2 for remote communication
     * @return std::shared_ptr<Descriptor2>
     */
    template <typename T>
    static std::shared_ptr<Descriptor2> create_from_value(T value, data_plane::DataPlaneResources2& data_plane_resources);

    /**
     * @brief Creates a Descriptor2 instance from a byte stream
     *
     * @param view byte stream
     * @param data_plane_resources reference to DataPlaneResources2 for remote communication
     * @return std::shared_ptr<Descriptor2>
     */
    static std::shared_ptr<Descriptor2> create_from_bytes(memory::buffer_view&& view, data_plane::DataPlaneResources2& data_plane_resources);

    /**
     * @brief Fetches all deferred payloads from the sending remote machine
     */
    void fetch_remote_payloads();

  protected:
    Descriptor2(std::any value, data_plane::DataPlaneResources2& data_plane_resources):
        m_value(value), m_data_plane_resources(data_plane_resources) {}
    Descriptor2(std::unique_ptr<codable::DescriptorObjectHandler> encoded_object, data_plane::DataPlaneResources2& data_plane_resources):
        m_encoded_object(std::move(encoded_object)), m_data_plane_resources(data_plane_resources) {}

    void setup_remote_payloads();

    std::any m_value;

    std::vector<memory::buffer> m_local_buffers;

    std::unique_ptr<codable::DescriptorObjectHandler> m_encoded_object;

    data_plane::DataPlaneResources2& m_data_plane_resources;
};

/**
 * @brief Class used for type erasure of Descriptor2 when serialized with class T instance
 */
template <typename T>
class TypedDescriptor : public Descriptor2
{
  public:
    codable::protos::DescriptorObject& encoded_object()
    {
        // If the encoded object does not exist yet, lazily create it
        if (!m_encoded_object)
        {
            m_encoded_object = std::move(mrc::codable::encode2<T>(std::any_cast<const T&>(m_value)));
        }

        return Descriptor2::encoded_object();
    }

  private:
    template <typename U>
    friend std::shared_ptr<Descriptor2> Descriptor2::create_from_value(U value, data_plane::DataPlaneResources2& data_plane_resources);

    // Private constructor to prohibit instantiation of this class outside of use in create_from_value
    TypedDescriptor(T value, data_plane::DataPlaneResources2& data_plane_resources):
        Descriptor2(std::move(value), data_plane_resources) {}
};

template <typename T>
std::shared_ptr<Descriptor2> Descriptor2::create_from_value(T value, data_plane::DataPlaneResources2& data_plane_resources)
{
    return std::shared_ptr<Descriptor2>(new TypedDescriptor<T>(std::move(value), data_plane_resources));
}

template <typename T>
[[nodiscard]] const T Descriptor2::deserialize()
{
    T return_value = m_value.has_value() ? std::move(std::any_cast<T>(m_value)) :
                                           std::move(mrc::codable::decode2<T>(*m_encoded_object));
    m_value.reset();
    return std::move(return_value);
}
}  // namespace mrc::runtime

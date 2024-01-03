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
#include "mrc/memory/memory_block_provider.hpp"
#include "mrc/type_traits.hpp"  // IWYU pragma: keep
#include "mrc/utils/macros.hpp"

#include <glog/logging.h>

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

class ValueDescriptor
{
  private:
    virtual std::unique_ptr<codable::EncodedObjectWithPayload> encode(
        std::shared_ptr<memory::memory_block_provider> block_provider) = 0;

    friend LocalDescriptor2;
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
    static std::unique_ptr<TypedValueDescriptor<T>> from_local(std::unique_ptr<LocalDescriptor2> local_descriptor)
    {
        T temp;

        // Perform a decode to get the value

        return std::unique_ptr<TypedValueDescriptor<T>>(new TypedValueDescriptor<T>(std::move(temp)));
    }

  private:
    TypedValueDescriptor(T&& value) : m_value(std::move(value)) {}

    std::unique_ptr<codable::EncodedObjectWithPayload> encode(
        std::shared_ptr<memory::memory_block_provider> block_provider) override
    {
        return mrc::codable::encode2(m_value, block_provider);
    }

    T m_value;
};

// Combines a EncodedObjectProto with a local registered buffers if needed
class LocalDescriptor2
{
  public:
    codable::EncodedObjectProto& encoded_object() const;

    static std::unique_ptr<LocalDescriptor2> from_value(std::unique_ptr<ValueDescriptor> value_descriptor,
                                                        std::shared_ptr<memory::memory_block_provider> block_provider);

    static std::unique_ptr<LocalDescriptor2> from_remote(std::unique_ptr<RemoteDescriptor2> remote_descriptor,
                                                         data_plane::DataPlaneResources2& data_plane_resources);

  private:
    LocalDescriptor2(std::unique_ptr<codable::EncodedObjectWithPayload> encoded_object,
                     std::unique_ptr<ValueDescriptor> value_descriptor = nullptr);

    std::unique_ptr<codable::EncodedObjectWithPayload> m_encoded_object;

    std::unique_ptr<ValueDescriptor> m_value_descriptor;  // Necessary to keep the value alive when serializing
};

class RemoteDescriptor2
{
  public:
    codable::EncodedObjectProto& encoded_object() const;

    static std::unique_ptr<RemoteDescriptor2> from_encoded_object(
        std::unique_ptr<codable::EncodedObjectProto> encoded_object);

  private:
    std::unique_ptr<codable::EncodedObjectProto> m_encoded_object;
};

}  // namespace mrc::runtime

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
#include "mrc/codable/encode.hpp"
#include "mrc/memory/buffer.hpp"
#include "mrc/memory/buffer_view.hpp"

#include <memory>
#include <utility>

namespace mrc::codable {

class EncodedStorage
{
  public:
    explicit EncodedStorage(std::unique_ptr<mrc::codable::IDecodableStorage> encoding);
    virtual ~EncodedStorage() = default;

  private:
    std::unique_ptr<mrc::codable::IDecodableStorage> m_encoding;
};

template <typename T>
class EncodedObject final : public EncodedStorage
{
    EncodedObject(T&& object, std::unique_ptr<mrc::codable::IDecodableStorage> encoding) :
      EncodedStorage(std::move(encoding)),
      m_object(std::move(object))
    {}

  public:
    ~EncodedObject() final = default;

    static std::unique_ptr<EncodedObject<T>> create(T&& object, std::unique_ptr<mrc::codable::ICodableStorage> storage)
    {
        mrc::codable::encode(object, *storage);
        return std::unique_ptr<EncodedObject<T>>(new EncodedObject(std::move(object), std::move(storage)));
    }

  private:
    T m_object;
};

template <typename T>
class EncodedObject<std::unique_ptr<T>> : public EncodedStorage
{
    EncodedObject(std::unique_ptr<T> object, std::unique_ptr<mrc::codable::IDecodableStorage> encoding) :
      EncodedStorage(std::move(encoding)),
      m_object(std::move(object))
    {}

  public:
    static std::unique_ptr<EncodedObject<T>> create(std::unique_ptr<T> object,
                                                    std::unique_ptr<mrc::codable::ICodableStorage> storage)
    {
        mrc::codable::encode(*object, *storage);
        return std::unique_ptr<EncodedObject<T>>(new EncodedObject(std::move(object), std::move(storage)));
    }

  private:
    std::unique_ptr<T> m_object;
};

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
    void add_eager_descriptor(memory::const_buffer_view view);

    // Adds a remote memory descriptor and sets the properties
    void add_remote_memory_descriptor(uint64_t instance_id,
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

}  // namespace mrc::codable

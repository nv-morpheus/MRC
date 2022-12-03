/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <utility>

namespace mrc::codable {

class EncodedStorage
{
  public:
    explicit EncodedStorage(std::unique_ptr<mrc::codable::IDecodableStorage> encoding);
    virtual ~EncodedStorage() = default;

    IDecodableStorage& encoding() const;

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

}  // namespace mrc::codable

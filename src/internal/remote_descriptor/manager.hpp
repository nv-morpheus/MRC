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

#include "internal/remote_descriptor/remote_descriptor.hpp"
#include "internal/remote_descriptor/storage.hpp"

#include <map>

namespace srf::internal::remote_descriptor {

class Manager final : public std::enable_shared_from_this<Manager>
{
  public:
    template <typename T>
    RemoteDescriptor register_object(T&& object)
    {
        return store_object(TypedStorage<T>::create(std::move(object)));
    }

    std::size_t size() const;

  private:
    RemoteDescriptor store_object(std::unique_ptr<Storage> object);

    void decrement_tokens(std::size_t object_id, std::size_t token_count);

    std::map<std::size_t, std::unique_ptr<Storage>> m_stored_objects;

    friend RemoteDescriptor;
};

}  // namespace srf::internal::remote_descriptor

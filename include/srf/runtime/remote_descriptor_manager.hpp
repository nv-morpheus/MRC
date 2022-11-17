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

#include "srf/codable/api.hpp"
#include "srf/codable/encoded_object.hpp"
#include "srf/protos/codable.pb.h"
#include "srf/runtime/remote_descriptor.hpp"

#include <memory>

namespace srf::runtime {

class RemoteDescriptor;
class IRemoteDescriptorHandle;

/**
 * @brief Publid interface of the RemoteDescriptorManager
 *
 */
class IRemoteDescriptorManager
{
  public:
    virtual ~IRemoteDescriptorManager() = default;

    template <typename T>
    RemoteDescriptor register_object(T&& object)
    {
        return register_encoded_object(codable::EncodedObject<T>::create(std::move(object), create_storage()));
    }

    virtual RemoteDescriptor register_encoded_object(std::unique_ptr<codable::EncodedStorage> object) = 0;

  protected:
    virtual std::unique_ptr<codable::ICodableStorage> create_storage()           = 0;
    virtual void release_handle(std::unique_ptr<IRemoteDescriptorHandle> handle) = 0;

    friend RemoteDescriptor;
};

}  // namespace srf::runtime

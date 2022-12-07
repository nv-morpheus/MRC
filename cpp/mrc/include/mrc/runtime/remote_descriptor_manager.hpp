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

#include "mrc/codable/api.hpp"
#include "mrc/codable/encoded_object.hpp"
#include "mrc/runtime/remote_descriptor.hpp"

#include <memory>

namespace mrc::runtime {

class IRemoteDescriptorHandle;

/**
 * @brief Public interface of the RemoteDescriptorManager
 *
 * A RemoteDescriptor manager is a partition-level runtime resource.
 */
class IRemoteDescriptorManager
{
  public:
    virtual ~IRemoteDescriptorManager() = default;

    /**
     * @brief Take ownership of an object T and provide back a RemoteDescriptor
     *
     * The remote descriptor manager will take ownership of a given object T, encode the description of that object in
     * an encoded storage object, then provide back a RemoteDescriptor object which is used to manage the lifecycle of
     * the original object T as well as decoding a copy of the original object. RemoteDescriptors can be transferred
     * across the MRC data plane to other physical machines/processes. When being decoded on a remote instance, the MRC
     * data plane uses the UCX communications library which will optimize the network transport based on the available
     * hardware on the source and destination machines.
     *
     * @tparam T
     * @param object
     * @return RemoteDescriptor
     */
    template <typename T>
    RemoteDescriptor register_object(T&& object)
    {
        return register_encoded_object(codable::EncodedObject<T>::create(std::move(object), create_storage()));
    }

    /**
     * @brief Take ownership of an EncodedStorage object and provide back a RemoteDescriptor
     *
     * Similar to `register_object`, but the backing object has already been captured and encoded by the EncodedStorage
     * object.
     *
     * @param object
     * @return RemoteDescriptor
     */
    virtual RemoteDescriptor register_encoded_object(std::unique_ptr<codable::EncodedStorage> object) = 0;

  protected:
    // Provides a ICodableStorage backed by the partition resources
    virtual std::unique_ptr<codable::ICodableStorage> create_storage() = 0;

    // Release the IRemoteDescriptorHandle by decrementing the *global* reference counting tokens by the number of
    // tokens held by the handle. This method will trigger *yielding* network communication if instance_id of the handle
    // is different from the instance_id of the remote descriptor manager.
    //
    // todo(mdemoret/ryanolson) - consider renaming to await_release_handle to indicate that the method may yield the
    // execution context
    virtual void release_handle(std::unique_ptr<IRemoteDescriptorHandle> handle) = 0;

    friend RemoteDescriptor;
};

}  // namespace mrc::runtime

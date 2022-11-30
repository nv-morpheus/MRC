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
#include "mrc/runtime/remote_descriptor_handle.hpp"  // IWYU pragma: keep
#include "mrc/utils/macros.hpp"

#include <glog/logging.h>

#include <cstddef>
#include <memory>

namespace mrc::internal::remote_descriptor {
class Manager;
}  // namespace mrc::internal::remote_descriptor

namespace mrc::runtime {

// class IRemoteDescriptorHandle;
class IRemoteDescriptorManager;

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
class RemoteDescriptor final
{
  public:
    RemoteDescriptor() = default;
    ~RemoteDescriptor();

    DELETE_COPYABILITY(RemoteDescriptor);
    DEFAULT_MOVEABILITY(RemoteDescriptor);

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
    T decode(std::size_t object_idx = 0)
    {
        CHECK(m_handle);
        return codable::Decoder<T>(*m_handle).deserialize(object_idx);
    }

    /**
     * @brief Releases the RemoteDescriptor causing a decrement of the global token count
     */
    void release_ownership();

    /**
     * @brief Returns true if this object is still connected to the global object; otherwise, ownership has been
     * transferred or released.
     *
     * @return true
     * @return false
     */
    bool has_value() const;

    /**
     * @brief Returns true if this object is still connected to the global object; otherwise, ownership has been
     * transferred or released.
     *
     * @return true
     * @return false
     */
    operator bool() const
    {
        return has_value();
    }

  private:
    RemoteDescriptor(std::shared_ptr<IRemoteDescriptorManager> manager,
                     std::unique_ptr<IRemoteDescriptorHandle> handle);

    std::unique_ptr<IRemoteDescriptorHandle> release_handle();

    std::shared_ptr<IRemoteDescriptorManager> m_manager;
    std::unique_ptr<IRemoteDescriptorHandle> m_handle;

    friend internal::remote_descriptor::Manager;
};

}  // namespace mrc::runtime

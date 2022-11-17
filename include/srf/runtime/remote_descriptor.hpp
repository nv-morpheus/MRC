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

#include "srf/codable/api.hpp"
#include "srf/runtime/remote_descriptor_handle.hpp"
#include "srf/utils/macros.hpp"

#include <memory>

namespace srf::internal::remote_descriptor {
class Manager;
}  // namespace srf::internal::remote_descriptor

namespace srf::runtime {

class IRemoteDescriptorManager;

class RemoteDescriptor final
{
  public:
    RemoteDescriptor() = default;
    ~RemoteDescriptor();

    DELETE_COPYABILITY(RemoteDescriptor);
    DEFAULT_MOVEABILITY(RemoteDescriptor);

    bool has_value() const;

    operator bool() const
    {
        return has_value();
    }

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

  private:
    RemoteDescriptor(std::shared_ptr<IRemoteDescriptorManager> manager,
                     std::unique_ptr<IRemoteDescriptorHandle> handle);

    std::unique_ptr<IRemoteDescriptorHandle> release_handle();

    std::shared_ptr<IRemoteDescriptorManager> m_manager;
    std::unique_ptr<IRemoteDescriptorHandle> m_handle;

    friend internal::remote_descriptor::Manager;
};

}  // namespace srf::runtime

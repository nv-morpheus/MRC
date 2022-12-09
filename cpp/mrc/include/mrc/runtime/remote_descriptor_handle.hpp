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

namespace mrc::codable::protos {
class RemoteDescriptor;
}

namespace mrc::runtime {

/**
 * @brief An IDecodableStorage object that owns the object encoding, backing instance_id, and reference counting tokens,
 * but does *not* own the backing object.
 *
 * This object can be decoded (using the resources of the backing partition), separated from the RemoteDescriptor and
 * used to transport and recreate the RemoteDescriptor an another machine or can be globally released by the manager.
 */
struct IRemoteDescriptorHandle : public virtual codable::IDecodableStorage
{
    ~IRemoteDescriptorHandle() override = default;

    virtual const mrc::codable::protos::RemoteDescriptor& remote_descriptor_proto() const = 0;
};

}  // namespace mrc::runtime

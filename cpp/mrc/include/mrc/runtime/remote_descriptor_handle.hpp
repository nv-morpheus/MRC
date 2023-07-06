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
#include "mrc/protos/codable.pb.h"
#include "mrc/utils/macros.hpp"

#include <cstdint>
#include <memory>

namespace mrc::codable::protos {
class RemoteDescriptor;
}

namespace mrc::remote_descriptor {
class Manager;
}  // namespace mrc::remote_descriptor

namespace mrc::runtime {

class LocalDescriptorHandle
{
  public:
    LocalDescriptorHandle() = default;
    LocalDescriptorHandle(std::unique_ptr<mrc::codable::protos::RemoteDescriptor> proto) : m_proto(std::move(proto)) {}

    DEFAULT_MOVEABILITY(LocalDescriptorHandle);
    DELETE_COPYABILITY(LocalDescriptorHandle);

    virtual ~LocalDescriptorHandle() = default;

    std::unique_ptr<mrc::codable::protos::RemoteDescriptor> release()
    {
        return std::move(m_proto);
    }

    const mrc::codable::protos::RemoteDescriptor& remote_descriptor_proto() const;

  private:
    uint64_t m_storage_id;

    std::unique_ptr<mrc::codable::protos::RemoteDescriptor> m_proto;

    // remote_descriptor::Manager& m_manager;
};

class RemoteDescriptorHandle
{
  public:
    virtual ~RemoteDescriptorHandle() = default;

    const mrc::codable::protos::RemoteDescriptor& remote_descriptor_proto() const;

    std::unique_ptr<codable::IDecodableStorage> release();

  private:
    // uint64_t m_storage_id;
    std::unique_ptr<codable::IDecodableStorage> m_storage;

    // remote_descriptor::Manager& m_manager;
};

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

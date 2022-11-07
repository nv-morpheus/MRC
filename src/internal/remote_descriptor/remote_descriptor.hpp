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

#include "srf/codable/forward.hpp"

#include <memory>

namespace srf::internal::remote_descriptor {

class Manager;

/**
 * RemoteDescriptor is an object that points to and maintains a global reference count to an encoded object which owned
 * by a singular RemoteDescriptorManager within the distributed SRF runtime. The actual object maybe local or remote to
 * the instance holding the object.
 *
 * When a RemoteDescriptor is destructed, it triggers an atomic global decrement of the reference count of the object
 * equal to the number of tokens held by the RemoteDescriptor. When the global reference count goes to zero, the
 * remote_descriptor::Manager which owns the EncodedObject will destory the object and release the memory.
 */
class RemoteDescriptor final
{
    RemoteDescriptor(std::shared_ptr<Manager> manager, std::unique_ptr<srf::codable::protos::RemoteDescriptor> rd);

  public:
    ~RemoteDescriptor();

    operator bool() const
    {
        return bool(m_descriptor);
    }

    void release();

  private:
    std::unique_ptr<srf::codable::protos::RemoteDescriptor> m_descriptor;
    std::shared_ptr<Manager> m_manager;
    friend Manager;
};

}  // namespace srf::internal::remote_descriptor
